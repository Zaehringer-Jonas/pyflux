# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018


Scan Widget

@author:  Jonas Zaehringer
original template: Luciano A. Masullo


Status:
    
    yz scan missing
    bidirectional?
    FLIM not properly working
    

"""

import numpy as np
import time
from datetime import date, datetime
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import tools.tools as tools
import ctypes as ct
from PIL import Image
from PIL.TiffTags import TAGS
from tkinter import Tk, filedialog
import tifffile as tiff
import scipy.optimize as opt
import scipy.stats as ss

from threading import Thread


import owis_ps10.ps10 as owis # OD filter motor

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5 import QtTest
import qdarkstyle
from PyQt5.QtWidgets import QSlider, QMessageBox

import timeit
import queue

import tcspc

from pipython import GCSDevice, pitools
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
from tools.lineprofile import linePlotWidget
from tools.PSF import poly_func
from tools.PSF import poly_func2D
from tools.PSF import poly_func2Dpl

from nidaqmx import Task

π = np.pi


    
    
class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(dict)
    closeSignal = pyqtSignal()
    liveviewSignal = pyqtSignal(bool, str)
    frameacqSignal = pyqtSignal(bool)
    
    FitandMoveSignal = pyqtSignal(np.ndarray)
    PickandDestroySignal = pyqtSignal(np.ndarray)
    
    
    
   

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)



        self.roi = None
        self.lineROI = None
        self.plotlifetime = False
        self.EBPscatter = [None, None, None, None]
        self.EBPcenters = np.zeros((4, 2))
        self.advanced = False
        self.EBPshown = True
        self.lastpxsize = 500 # TODO: bad get to backend
        self.fitting = False
        self.plotlifetime = False # maybe not needed to init..
        self.plotdualcolor = False
        self.image = np.zeros((128, 128))
        self.lastScanMode = 'xz'
        self.initialDir = r'D:\Data'
        self.ODinitflag = False
        # set up GUI

        self.setup_gui()
                
        # connections between changes in parameters and emit_param function
        
        self.NofPixelsEdit.textChanged.connect(self.emit_param)
        self.scanRangeEdit.textChanged.connect(self.emit_param)
        self.pxTimeEdit.textChanged.connect(self.emit_param)
        self.initialPosEdit.textChanged.connect(self.emit_param)
        self.pulseBinEdit.textChanged.connect(self.emit_param)

        self.scanODEdit.textChanged.connect(self.emit_param)
        self.tcspcODEdit.textChanged.connect(self.emit_param)
        self.lifetimeEdit.textChanged.connect(self.emit_param)
                

        
        self.waitingTimeEdit.textChanged.connect(self.emit_param)
        self.scanDirectionType.activated.connect(self.emit_param)
        self.scanColorType.activated.connect(self.emit_param)
        self.scanColorType.activated.connect(self.toggle_dualcolor)
        
        
        self.scanMode.activated.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.xStepEdit.textChanged.connect(self.emit_param)
        self.yStepEdit.textChanged.connect(self.emit_param)
        self.zStepEdit.textChanged.connect(self.emit_param)
        self.moveToEdit.textChanged.connect(self.emit_param)
        self.intensityEdit.textChanged.connect(self.emit_intensity_edit)
        self.offsetEdit.textChanged.connect(self.emit_param)
        self.intensitySlider.valueChanged.connect(self.emit_intensity_slider)
        self.midbeamdropdown.activated.connect(self.emit_param)
        
        
        self.FLIM.activated.connect(self.toggle_lifetime)
        
    def emit_intensity_slider(self):
        val = self.intensitySlider.value()      
        self.intensityEdit.setText('{}'.format(np.around(val/10, 1)))
        self.emit_param()
        
    def emit_intensity_edit(self):
        
        try:
            val = int(float(self.intensityEdit.text())*10)
            if (val >= 0) and (val <= 40):
                self.intensitySlider.setValue(val)
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[Scan] Warning", "Input error. Please check scan inputs.")
#        self.emit_param()
        
        
    def emit_param(self):
        
        params = dict()
        
        try:
            
            initpos = np.array(self.initialPosEdit.text().split(' '), dtype=np.float64)
            scrange = float(self.scanRangeEdit.text())
            
            
            if self.scanMode.currentText() == "xy":
                if (initpos[0] + scrange > 100 ) or (initpos[1] + scrange > 100 ):
                    QMessageBox.about(self, "[Scan] Warning", "Out of Range!!")
            else:
                if (initpos[2] + scrange > 10 ):
                    QMessageBox.about(self, "[Scan] Warning", "Out of Range!!")
                
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[Scan] Warning", "Input error. Please check scan inputs.")
        
        
        
        
        
        try:
            params['scanDirectionType'] = self.scanDirectionType.currentText()
            params['scancolor'] = self.scanColorType.currentText()
            
            
            params['scanType'] = self.scanMode.currentText()
            params['scanRange'] = float(self.scanRangeEdit.text())
            
            if int(self.NofPixelsEdit.text()) != 0:
                params['NofPixels'] = int(self.NofPixelsEdit.text())
            else:
                params['NofPixels'] = 1
            params['pxTime'] = float(self.pxTimeEdit.text())
            params['initialPos'] = np.array(self.initialPosEdit.text().split(' '),
                                            dtype=np.float64)
            params['pulse_bin'] = np.array(self.pulseBinEdit.text().split(' '),
                                                  dtype=np.float32)
            
            params['waitingTime'] = int(self.waitingTimeEdit.text())  # in µs
            params['fileName'] = os.path.join(self.folderEdit.text(),
                                              self.filenameEdit.text())
            params['moveToPos'] = np.array(self.moveToEdit.text().split(' '),
                                           dtype=np.float16)
            
            params['xStep'] = float(self.xStepEdit.text())
            params['yStep'] = float(self.yStepEdit.text())
            params['zStep'] = float(self.zStepEdit.text())
            params['intensitySlider'] = float(self.intensitySlider.value())
            params['intensityOffset'] = float(self.offsetEdit.text())
            params['midbeam'] = self.midbeamdropdown.currentText()
            params['lifetimeflag'] = self.plotlifetime
    
            params['scanOD'] = float(self.scanODEdit.text())
            params['tcspcOD'] = float(self.tcspcODEdit.text())
            params['lifetimelimit'] = float(self.lifetimeEdit.text())
            self.paramSignal.emit(params)
        
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[Scan] Warning", "Input error. Please check scan inputs.")
        
        
        #        params['Nframes'] = float(self.NframesEdit.text())

        
#        print("emit")
        
    @pyqtSlot(dict)
    def get_backend_param(self, params):
        
        frameTime = params['frameTime']
        pxSize = params['pxSize']
        maxCounts = params['maxCounts']
        initialPos = np.round(params['initialPos'], 3)
        
        if self.ODinitflag == False:
            ODangleoffsetinit = params['ODangleoffsetinit'] 
            self.offsetEdit.setText('{}'.format(np.around(ODangleoffsetinit, 2)))
            
            self.ODinitflag = True
        
        
        

        
        self.frameTimeValue.setText('{}'.format(np.around(frameTime, 2)))
        self.pxSizeValue.setText('{}'.format(np.around(1000 * pxSize, 3))) # in nm
        self.maxCountsValue.setText('{}'.format(maxCounts)) 
        # self.initialPosEdit.setText('{} {} {}'.format(*initialPos))
        
        self.pxSize = pxSize
     
    @pyqtSlot(np.ndarray)
    def get_image(self, image):
        
        
#        print(image)
        
        if self.plotlifetime == False and self.plotdualcolor == False :
#            print(image.shape)
            if image.shape[-1] != 4:
                    
                
                self.img.setImage(image, autoLevels=True)
                       
                self.image = image
            
        if self.plotlifetime == True:
#            print("getimage", image.ndim)
#            print(image[...,1])
            
            #check dimension:
            
#            print("pl lifetime RGBA ",image.shape)
            if image.shape[-1] ==3:
                self.img.setImage(image, autoLevels=False, lut = None, levelMode  = "rgb")#, useRGBA = True) # 
                self.image = image
                
                self.img.setLevels([[0,1],[0,1],[0,1]])
            
        if self.plotdualcolor == True:
#            print("pl dual color RGBA ",image.shape)
            if image.shape[-1] == 4:
                self.img.setImage(image, autoLevels=False, lut = None, levelMode  = "rgba", useRGBA = True) # 
                self.image = image
                
                self.img.setLevels([[0,1],[0,1],[0,1],[0,1]])
            
#            
#    
#    def calculate_color_image(self):
#        #TODO make this working
#        
#        RGBAline = []
#        self.maxlifetime = 3.2
#        
#        lut = viewbox_tools.generatePgColormap(cmaps.parula)#
#    #            print(lifetimes)
#        color = lut.map(self.lifetimeimage / self.maxlifetime) # todochange!!! lut is not [2,2,2]
#        color = color.astype(np.float64)/255.
##            print(pxcount)
##            print(maxcount)
#
#        maxcount = 500
#        pxcount = self.lineData[-1]
#        if maxcount > 0:
#            
##                print("qauot", pxcount/ maxcount )
#            color[-1] =  pxcount/ maxcount #*255 # [0:1] ->  [0:255]
#        else: 
#            color[-1] = 0
#            
#        
##            print(color)
#        RGBAline.append(color)

    
    
    def releaseLiveviewButton(self):
        self.liveviewButton.setChecked(False)
        
    
    @pyqtSlot(bool)
    def toggleShutterButton(self, checked):
        
        if checked == False:
            self.liveviewButton.setChecked(False)    
            
        if checked == True:
            self.liveviewButton.setChecked(True)    
            
    
    
    def toggle_advanced(self):
        
        if self.advanced:
            
            self.pulseBinLabel.show()
            self.pulseBinEdit.show()
            self.waitingTimeLabel.show()
            self.waitingTimeEdit.show() 
            self.saveReltimesButton.show()
            
            self.midbeamlabel.show()
            self.midbeamdropdown.show()
            
            self.scanODLabel.hide()
            self.scanODEdit.hide()
            self.tcspcODLabel.hide()
            self.tcspcODEdit.hide()             
            self.lifetimeLabel.hide()
            self.lifetimeEdit.hide()     
            
            
            self.advanced = False
            
        else:
            
            self.pulseBinLabel.hide()
            self.pulseBinEdit.hide()
            self.waitingTimeLabel.hide()
            self.waitingTimeEdit.hide() 
            self.saveReltimesButton.hide()

            self.midbeamlabel.hide()
            self.midbeamdropdown.hide()
            
            
            self.scanODLabel.show()
            self.scanODEdit.show()
            self.tcspcODLabel.show()
            self.tcspcODEdit.show() 
            self.lifetimeLabel.show()
            self.lifetimeEdit.show()    
            
            self.advanced = True

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
    


    def toggle_liveview(self):
        ###############################################################################!!
        self.lastScanMode = self.scanMode.currentText()
        self.lastpxsize = self.pxSize
        
        print(self.lastScanMode)
        if self.liveviewButton.isChecked(): 
            self.liveviewSignal.emit(True, 'liveview')
            
            if self.roi is not None:

                self.vb.removeItem(self.roi)
                self.roi.hide()
    
                self.ROIButton.setChecked(False)
            
            if self.lineROI is not None:

                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None

            else:
    
                pass

        else:
            self.liveviewSignal.emit(False, 'liveview')   
            
    def toggle_frame_acq(self):

        if self.acquireFrameButton.isChecked():
            self.frameacqSignal.emit(True)
            
            if self.roi is not None:

                self.vb.removeItem(self.roi)
                self.roi.hide()
    
                self.ROIButton.setChecked(False)
                self.liveviewButton.setChecked(False)
            
            if self.lineROI is not None:

                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None

            else:
    
                pass

        else:
            self.frameacqSignal.emit(False)   

    def line_profile(self):
        ###############################################################################!!
        if self.lineROI is None:
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
            self.vb.addItem(self.lineROI)
            
            self.lplotWidget.show()
            
        else:

            self.vb.removeItem(self.lineROI)
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
            self.vb.addItem(self.lineROI)
            
        self.lineROI.sigRegionChanged.connect(self.update_line_profile)
        
    def update_line_profile(self):
        
        data = self.lineROI.getArrayRegion(self.image, self.img)
        self.lplotWidget.linePlot.clear()
        x = self.pxSize * np.arange(np.size(data))*1000
        self.lplotWidget.linePlot.plot(x, data)

        
    def toggle_ROI(self):
        self.select_ROIButton.setStyleSheet("background-color: red")
        ROIpen = pg.mkPen(color='r')
        npixels = int(self.NofPixelsEdit.text())
        ###############################################################################!!
        currentScanMode = self.scanMode.currentText()
        

        if self.lastScanMode == currentScanMode and self.roi is None or self.lastScanMode  == '':

            ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
         
        elif self.lastScanMode != currentScanMode and self.lineROI is None:
            
            
            
            
            
            self.lineROI = pg.LineSegmentROI([[0, 0.5 * npixels], [npixels,0.5 * npixels]], pen='r')
            self.vb.addItem(self.lineROI)

            
        elif self.roi is not None:
            #TODO add line case
            self.vb.removeItem(self.roi)
            self.roi.hide()

            ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
        if self.EBProiButton.isChecked:
            self.EBProiButton.setChecked(False)
            
    def select_ROI(self):
        ###############################################################################!! TO BACKEND?
        self.select_ROIButton.setStyleSheet("background-color: green")
        self.liveviewSignal.emit(False, 'liveview')
        currentScanMode = self.scanMode.currentText()
        scanrangeold = float(self.scanRangeEdit.text())
        print(currentScanMode, " current,  last ", self.lastScanMode )
        initialPos = np.array(self.initialPosEdit.text().split(' '),
                              dtype=np.float64)
        if self.roi is not None:
            if self.lastScanMode == currentScanMode or self.lastScanMode  == '':
                ROIsize = np.array(self.roi.size())
                ROIpos = np.array(self.roi.pos())
                
                npixels = int(self.NofPixelsEdit.text())
                pxSize = self.lastpxsize
                
        
                newPos_px = tools.ROIscanRelativePOS(ROIpos, npixels, ROIsize[1])
                if currentScanMode == 'xy':
                    
                    
                    newPos_µm = newPos_px * pxSize + initialPos[0:2] 
                    
                    newPos_µm = np.around(newPos_µm, 3)
                    newRange_px = ROIsize[1]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 2)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                              newPos_µm[1],
                                                              initialPos[2]))
                if currentScanMode == 'xz':
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[0],initialPos[2] ])
                    
                    newPos_µm = np.around(newPos_µm, 3)
                    newRange_px = ROIsize[1]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 3)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                              initialPos[1],
                                                              newPos_µm[1]))
                    
                    
                    
                if currentScanMode == 'yz':
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[1],initialPos[2] ])
                    
                    newPos_µm = np.around(newPos_µm, 3)
                    newRange_px = ROIsize[1]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 3)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(initialPos[1],
                                                                newPos_µm[0],
                                                              newPos_µm[1]))
                
    
            
            else:
                
                ROIpos = np.array([handle['pos'] + self.lineROI.state['pos'] for handle in self.lineROI.handles])
                npixels = int(self.NofPixelsEdit.text())
                pxSize = self.lastpxsize
                print(ROIpos)
                scanrangeold = float(self.scanRangeEdit.text())
                newPos_px =  np.array(ROIpos[0])
    
    
    
                if currentScanMode == 'xy' and self.lastScanMode == 'xz':
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[0],initialPos[2] - scanrangeold/2.])
    
    
                    newPos_µm = np.around(newPos_µm, 3)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 2)
                   
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                          initialPos[1]- scanrangeold/2.,
                                                          newPos_µm[1]
                                                          ))
    
                if currentScanMode == 'xz' and self.lastScanMode == 'xy':
                    
                    
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[0],initialPos[1]])
        
                    newPos_µm = np.around(newPos_µm, 3)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 2)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                          newPos_µm[1],
                                                          initialPos[2]))
    
                if currentScanMode == 'yz' and self.lastScanMode == 'xy':
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[0],initialPos[1]])
        
                    newPos_µm = np.around(newPos_µm, 3)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 2)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                          newPos_µm[1],
                                                          initialPos[2]))
                
                
                if currentScanMode == 'yz' and self.lastScanMode == 'xz':
                    
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[0],initialPos[2]  - scanrangeold/2.])
        
                    newPos_µm = np.around(newPos_µm, 2)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 3)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                          initialPos[1],
                                                          newPos_µm[1]))
                    
                    
                if currentScanMode == 'xz' and self.lastScanMode == 'yz':
                    
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[1],initialPos[2]  - scanrangeold/2.])
        
                    newPos_µm = np.around(newPos_µm, 2)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 3)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(initialPos[0],
                                                                newPos_µm[0],
                                                                newPos_µm[1]))
                    
                    
                if currentScanMode == 'xy' and self.lastScanMode == 'yz':
                    
                    
                    
                    newPos_µm = newPos_px * pxSize + np.array([initialPos[1],initialPos[2]  - scanrangeold/2.])
        
                    newPos_µm = np.around(newPos_µm, 2)
                 
                    
                    newRange_px = ROIpos[1,0] - ROIpos[0,0]
                    newRange_µm = pxSize * newRange_px
                    newRange_µm = np.around(newRange_µm, 3)
                    
                    
                    self.initialPosEdit.setText('{} {} {}'.format(initialPos[0],
                                                                newPos_µm[0],
                                                                newPos_µm[1]))
                
                
                
            self.scanRangeEdit.setText('{}'.format(newRange_µm))
        
            self.emit_param()
    def toggle_dualcolor(self, val):
        
        scancolor = self.scanColorType.currentText()
               
        if scancolor == 'green/red':
            print("Toggle dual color")
            self.plotdualcolor = True
        
        elif scancolor != 'green/red' and self.plotdualcolor == True:
            print("toggle mono color")
            self.image = np.zeros((128, 128))           
            self.img.setImage(np.zeros((128, 128)), autoLevels=True) #todo make smart!
            self.plotdualcolor = False
            lut = viewbox_tools.generatePgColormap(cmaps.parula)
            self.hist.gradient.setColorMap(lut)
            
            for tick in self.hist.gradient.ticks:
                tick.hide()
#            self.hist.vb.setLimits(yMin=0, yMax=10000)
        self.emit_param()
            
        if scancolor == 'green':
            self.midbeamdropdown.setCurrentIndex(0)
            self.emit_param()
        if scancolor == 'red':
            self.midbeamdropdown.setCurrentIndex(3)
            self.emit_param()
        




    def toggle_lifetime(self, val):
#        print("toggle lf", val)




        FLIMtype = self.FLIM.currentText()
        if FLIMtype == "FLIM":
            print("Toggle FLIM")
            self.plotlifetime = True
#            self.image = np.zeros((128, 128, 4)) 
#            self.img.setImage(np.zeros((128, 128)))
#            self.img.setLevels([[0,1],[0,1],[0,1],[0,1]])
            
            lut = viewbox_tools.generatePgColormap(cmaps.parula)
            self.hist.gradient.setColorMap(lut)
            for tick in self.hist.gradient.ticks:
                tick.hide()
            
#            print("Toggle FLIM")#           
# lut = viewbox_tools.generatePgColormap(cmaps.parula)#
#            self.hist.gradient.setColorMap(lut)
#            self.hist.vb.setLimits(yMin=0, yMax=10000)
            self.emit_param()
            
        if FLIMtype == "Scan":
            print("toggle scan")
            self.image = np.zeros((128, 128))           
            self.img.setImage(np.zeros((128, 128)), autoLevels=True) #todo make smart!
            self.plotlifetime = False
            lut = viewbox_tools.generatePgColormap(cmaps.parula)
            self.hist.gradient.setColorMap(lut)
            
            for tick in self.hist.gradient.ticks:
                tick.hide()
#            self.hist.vb.setLimits(yMin=0, yMax=10000)
            self.emit_param()
            
#            self.hist.hide()

        
    def moveto_crosshair(self):
        self.crosshairCheckbox.setChecked(False)
        
        xcenter = int(np.round(self.ch.vLine.value(), 0))
        ycenter = int(np.round(self.ch.hLine.value(), 0))
        
        print(xcenter)
        self.FitandMoveSignal.emit(np.array([xcenter,ycenter]))

    @pyqtSlot(np.ndarray)
    def center_Crosshair(self, position):
        if self.crosshairCheckbox.isChecked() == False:
                
                self.crosshairCheckbox.setChecked(True)
#                self.ch.toggle()
                
        print("center CR:", str(position))
        
        
        
        self.ch.mouseClicked()
        self.ch.vLine.setPos(position[0])
        self.ch.hLine.setPos(position[1])
#        pass
        
        
        
        
        
        
    def set_EBP(self):
        
        pxSize = self.pxSize
        ROIsize = np.array(self.roi.size())
        
        for i in range(4):
        
            if self.EBPscatter[i] is not None:
                
                self.vb.removeItem(self.EBPscatter[i])
        
#        array = self.roi.getArrayRegion(self.scworker.image, self.img)
        ROIsize = np.array(self.roi.size())
        ROIpos_µm = np.array(self.roi.pos()) * pxSize
            
        xmin = ROIpos_µm[0]
        xmax = ROIpos_µm[0] + ROIsize[0] * pxSize
        
        ymin = ROIpos_µm[1]
        ymax = ROIpos_µm[1] + ROIsize[1] * pxSize
        
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2
        
        if self.EBPtype.currentText() == 'triangle':
        
            L = int(self.LEdit.text())/1000 # in µm
            θ = π * np.array([1/6, 5/6, 3/2])
            ebp = (L/2)*np.array([[0, 0], [np.cos(θ[0]), np.sin(θ[0])], 
                                 [np.cos(θ[1]), np.sin(θ[1])], 
                                 [np.cos(θ[2]), np.sin(θ[2])]])
            
            self.EBP = (ebp + np.array([x0, y0]))/pxSize
                                       
        print('[scan] EBP px', self.EBP)
            
        for i in range(4):
        
            if i == 0:
                mybrush = pg.mkBrush(255, 255, 0, 255)
                
            if i == 1:
                mybrush = pg.mkBrush(255, 127, 80, 255)
                
            if i == 2:
                mybrush = pg.mkBrush(135, 206, 235, 255)
                
            if i == 3:
                mybrush = pg.mkBrush(0, 0 ,0 , 255)
                
            self.EBPscatter[i] = pg.ScatterPlotItem([self.EBP[i][0]], 
                                                    [self.EBP[i][1]], 
                                                    size=10,
                                                    pen=pg.mkPen(None), 
                                                    brush=mybrush)            

            self.vb.addItem(self.EBPscatter[i])
        
        self.set_EBPButton.setChecked(False)
        
    def toggle_EBP(self):
        
        if self.EBPshown:
        
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.removeItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = False
            
        else:
            
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.addItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = True
    
        self.showEBPButton.setChecked(False)
        
        
    def toggle_PickandDestroy(self):
        clicks = []
        
        while not self.PickAndDestroyButton.isChecked():
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN:
                    clicks.append(event.pos)
                    print(clicks)

        print("done with clicking", clicks  )
        
        #Start Backend. Pick and Destroy
        
        #self.PickandDestroySignal.emit(clicks)
        
        
        
    def setup_gui(self):
                
        # image widget set-up and layout

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
        
        # widget with scanning parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        scanParamTitle = QtGui.QLabel('<h2><strong>Scan settings</strong></h2>')
        scanParamTitle.setTextFormat(QtCore.Qt.RichText)

        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('confocal scan')
        self.liveviewButton.setFont(QtGui.QFont('Helvetica', weight=QtGui.QFont.Bold))
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setStyleSheet("font-size: 12px; background-color:rgb(180, 180, 180)")
        self.liveviewButton.clicked.connect(self.toggle_liveview)
        self.liveviewButton.setStyleSheet("QPushButton:checked {background-color: rgb(255, 182, 193);}")
        
        # ROI buttons

        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.toggle_ROI)

        self.select_ROIButton = QtGui.QPushButton('select ROI')
        self.select_ROIButton.clicked.connect(self.select_ROI)
      
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Shutter open/close')
        self.shutterButton.setCheckable(True)
        

        self.shutterButton.setStyleSheet("QPushButton:checked {background-color: rgb(0, 255, 0);}")
        
        
        # Flipper button
        
        self.flipperButton = QtGui.QPushButton('Save TCSPC ')
        self.flipperButton.setCheckable(True)

        # Save Reltimes Button button
        
        self.saveReltimesButton = QtGui.QPushButton('Save Reltimes')
        self.saveReltimesButton.setCheckable(True)

        # Lifetime Button button
        
#        self.lifetimeButton = QtGui.QPushButton('Plot Lifetime')
#        
#        self.lifetimeButton.clicked.connect(lambda: self.toggle_lifetime(self.lifetimeButton.isChecked())) #TODO: check if 
#        self.lifetimeButton.setCheckable(True)

        #Select FLIM Dropdown Menu
        self.FLIM = QtGui.QComboBox()
        self.FLIMoptions = ['Scan', 'FLIM']
        self.FLIM.addItems(self.FLIMoptions)

      
        # Save current frame button

        self.currentFrameButton = QtGui.QPushButton('Save current frame')

        # preview scan button

        
        # move to center button
        
        self.moveToROIcenterButton = QtGui.QPushButton('Move to ROI center') 
#        self.moveToROIcenterButton.clicked.connect(self.select_ROI)

        # line profile button
        
        self.lineProfButton = QtGui.QPushButton('Line profile')
        self.lineProfButton.setCheckable(True)
        self.lineProfButton.clicked.connect(self.line_profile)


        # toggle crosshair
        
        self.crosshairCheckbox = QtGui.QCheckBox('Toggle Crosshair')
        self.crosshairCheckbox.stateChanged.connect(self.ch.toggle)
        
        # move to crosshair button
        
        self.crosshairButton = QtGui.QPushButton('Fit and Move to Center')
        self.crosshairButton.clicked.connect(self.moveto_crosshair)

        #Pick and Destroy Button
        
        self.PickAndDestroyButton = QtGui.QPushButton('Pick and Destroy')
        self.PickAndDestroyButton.setCheckable(True)
        self.PickAndDestroyButton.clicked.connect(self.toggle_PickandDestroy)




        
        # Scanning parameters

        self.initialPosLabel = QtGui.QLabel('Initial Pos'
                                            ' [x0, y0, z0] (µm)')
        self.initialPosEdit = QtGui.QLineEdit('30. 30. 5')
        self.scanRangeLabel = QtGui.QLabel('Scan range (µm)')
        self.scanRangeEdit = QtGui.QLineEdit('10')
        self.pxTimeLabel = QtGui.QLabel('Pixel time (µs)')
        self.pxTimeEdit = QtGui.QLineEdit('500')
        self.NofPixelsLabel = QtGui.QLabel('Number of pixels')
        self.NofPixelsEdit = QtGui.QLineEdit('80')
        
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeValue = QtGui.QLineEdit('')
        self.pxSizeValue.setReadOnly(True)
        self.frameTimeLabel = QtGui.QLabel('Frame time (s)')
        self.frameTimeValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        self.maxCountsLabel = QtGui.QLabel('Max counts per pixel')
        self.maxCountsValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        
        self.advancedButton = QtGui.QPushButton('Advanced options')
        self.advancedButton.setCheckable(True)
        self.advancedButton.clicked.connect(self.toggle_advanced)
        
        self.pulseBinLabel = QtGui.QLabel('Pulse Bin start')
        self.pulseBinEdit = QtGui.QLineEdit('0.1 12.6 25.2 37.3')
        self.waitingTimeLabel = QtGui.QLabel('Scan waiting time (µs)')
        self.waitingTimeEdit = QtGui.QLineEdit('0')

        self.scanODLabel = QtGui.QLabel('OD at scan')
        self.scanODEdit = QtGui.QLineEdit('3')
        self.tcspcODLabel = QtGui.QLabel('OD at tcspc')
        self.tcspcODEdit = QtGui.QLineEdit('2')


        self.lifetimeLabel = QtGui.QLabel('Lifetime Treshhold')
        self.lifetimeEdit = QtGui.QLineEdit('2.5')
        
        
        self.midbeamlabel = QtGui.QLabel('Middle Beam')
        self.midbeamdropdown = QtGui.QComboBox()
        self.beams = ['200', '455', '710', '965']
        self.midbeamdropdown.addItems(self.beams)
        
        self.toggle_advanced()

        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(120)
        self.fileWidget.setFixedWidth(230)

        # folder and buttons
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[scan] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[scan] Successfully created the directory {}'.format(folder))
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.load_folder)

        # scan selection

        self.scanModeLabel = QtGui.QLabel('Scan type')

        self.scanMode = QtGui.QComboBox()
        self.scanModes = ['xy', 'xz'] #, 'yz']
        self.scanMode.addItems(self.scanModes)
        
        self.scanDirectionType = QtGui.QComboBox()
        self.scanDirectionTypes = ['monodirectional', 'bidirectional'] 
        self.scanDirectionType.addItems(self.scanDirectionTypes)

        self.scanColorType = QtGui.QComboBox()
        self.scanColorTypes = ['green', 'red', 'green/red'] 
        self.scanColorType.addItems(self.scanColorTypes)
        
        self.scanDirectionType.hide()
        
        # widget with EBP parameters and buttons
        
        self.EBPWidget = QtGui.QFrame()
        self.EBPWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        EBPparamTitle = QtGui.QLabel('<h2><strong>Excitation Beam Pattern</strong></h2>')
        EBPparamTitle.setTextFormat(QtCore.Qt.RichText)
        
        self.EBProiButton = QtGui.QPushButton('EBP ROI')
        self.EBProiButton.setCheckable(True)
        self.EBProiButton.clicked.connect(self.toggle_ROI)
        
        self.showEBPButton = QtGui.QPushButton('show/hide EBP')
        self.showEBPButton.setCheckable(True)
        self.showEBPButton.clicked.connect(self.toggle_EBP)

        self.set_EBPButton = QtGui.QPushButton('set EBP')
        self.set_EBPButton.clicked.connect(self.set_EBP)
        
        # EBP selection

        self.EBPtypeLabel = QtGui.QLabel('EBP type')

        self.EBPtype = QtGui.QComboBox()
        self.EBPoptions = ['triangle', 'square']
        self.EBPtype.addItems(self.EBPoptions)
        
        self.Llabel = QtGui.QLabel('L (nm)')
        self.LEdit = QtGui.QLineEdit('100')
        
        # piezo navigation widget
        
        self.positioner = QtGui.QFrame()
        self.positioner.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.xUpButton = QtGui.QPushButton("(+x) ►")  # →
        self.xDownButton = QtGui.QPushButton("◄ (-x)")  # ←

        self.yUpButton = QtGui.QPushButton("(+y) ▲")  # ↑
        self.yDownButton = QtGui.QPushButton("(-y) ▼")  # ↓

        
        self.zUpButton = QtGui.QPushButton("(+z) ▲")  # ↑
        self.zDownButton = QtGui.QPushButton("(-z) ▼")  # ↓
        
        self.xStepLabel = QtGui.QLabel('x step (µm)')
        self.xStepEdit = QtGui.QLineEdit('0.020')
        
        self.yStepLabel = QtGui.QLabel('y step (µm)')
        self.yStepEdit = QtGui.QLineEdit('0.020')
        
        self.zStepLabel = QtGui.QLabel('z step (µm)')
        self.zStepEdit = QtGui.QLineEdit('0.020')
        
        # move to button

        self.moveToButton = QtGui.QPushButton('Move to')
        self.moveToLabel = QtGui.QLabel('Move to [x0, y0, z0] (µm)')
        self.moveToEdit = QtGui.QLineEdit('0 0 10')
        
        
        self.intensityWidget = QtGui.QFrame()
        self.intensityWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.intensityLabel = QtGui.QLabel('OD [0,4]')
        self.intensityEdit = QtGui.QLineEdit('3')
        
        self.offsetLabel = QtGui.QLabel('OD angle Offset')
        self.offsetEdit = QtGui.QLineEdit('0')
        
        # scan GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        dockArea = DockArea() 
        grid.addWidget(dockArea, 0, 0)
        
        EBPDock = Dock('EBP')
        EBPDock.setOrientation(o="vertical", force=True)
        EBPDock.updateStyle()
        EBPDock.addWidget(self.EBPWidget)
        dockArea.addDock(EBPDock)
        
        positionerDock = Dock('Positioner')
        positionerDock.setOrientation(o="vertical", force=True)
        positionerDock.updateStyle()
        positionerDock.addWidget(self.positioner)
        positionerDock.addWidget(self.intensityWidget)
        dockArea.addDock(positionerDock, 'above', EBPDock)
        
        paramDock = Dock('Scan parameters')
        paramDock.setOrientation(o="vertical", force=True)
        paramDock.updateStyle()
        paramDock.addWidget(self.paramWidget)
        paramDock.addWidget(self.fileWidget)
        dockArea.addDock(paramDock, 'above', positionerDock)
        
        imageDock = Dock('Confocal view')
        imageDock.addWidget(imageWidget)
        dockArea.addDock(imageDock, 'right', paramDock)
        
        # parameters widget layout

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        subgrid.addWidget(scanParamTitle, 0, 0, 1, 3)
        
        subgrid.addWidget(self.scanModeLabel, 2, 2)
        subgrid.addWidget(self.scanMode, 3, 2)
#        subgrid.addWidget(self.scanDirectionType, 4, 2)
        
        subgrid.addWidget(self.scanColorType, 4, 2)
        subgrid.addWidget(self.FLIM, 5, 2)
        subgrid.addWidget(self.liveviewButton, 6, 2)
        
        subgrid.addWidget(self.shutterButton, 8, 2)
        #subgrid.addWidget(self.flipperButton, 8, 2)
#        subgrid.addWidget(self.lifetimeButton, 8, 2)
        
        subgrid.addWidget(self.currentFrameButton, 9, 2)
        subgrid.addWidget(self.ROIButton, 10, 2)
        subgrid.addWidget(self.select_ROIButton, 11, 2)

        subgrid.addWidget(self.moveToROIcenterButton, 13, 2)
        subgrid.addWidget(self.lineProfButton, 14, 2)
        subgrid.addWidget(self.crosshairCheckbox, 15, 2)
        subgrid.addWidget(self.crosshairButton, 16, 2)
#        subgrid.addWidget(self.PickAndDestroyButton, 17, 2)

        subgrid.addWidget(self.initialPosLabel, 2, 0, 1, 2)
        subgrid.addWidget(self.initialPosEdit, 3, 0, 1, 2)
        subgrid.addWidget(self.scanRangeLabel, 4, 0)
        subgrid.addWidget(self.scanRangeEdit, 4, 1)
        subgrid.addWidget(self.pxTimeLabel, 5, 0)
        subgrid.addWidget(self.pxTimeEdit, 5, 1)
        subgrid.addWidget(self.NofPixelsLabel, 6, 0)
        subgrid.addWidget(self.NofPixelsEdit, 6, 1)
        
        subgrid.addWidget(self.pxSizeLabel, 7, 0)
        subgrid.addWidget(self.pxSizeValue, 7, 1)
        subgrid.addWidget(self.frameTimeLabel, 8, 0)
        subgrid.addWidget(self.frameTimeValue, 8, 1)
        subgrid.addWidget(self.maxCountsLabel, 9, 0)
        subgrid.addWidget(self.maxCountsValue, 9, 1)
        
        subgrid.addWidget(self.advancedButton, 10, 0)
        
        subgrid.addWidget(self.pulseBinLabel, 11, 0)
        subgrid.addWidget(self.pulseBinEdit, 12, 0)
        subgrid.addWidget(self.waitingTimeLabel, 13, 0)
        subgrid.addWidget(self.waitingTimeEdit, 14, 0)
        subgrid.addWidget(self.saveReltimesButton, 15, 0)
       
        
        subgrid.addWidget(self.scanODLabel, 11, 0)
        subgrid.addWidget(self.scanODEdit, 12, 0)
        subgrid.addWidget(self.tcspcODLabel, 13, 0)
        subgrid.addWidget(self.tcspcODEdit, 14, 0)
        
        subgrid.addWidget(self.lifetimeLabel, 15, 0)
        subgrid.addWidget(self.lifetimeEdit, 16, 0)
        
        
        subgrid.addWidget(self.midbeamlabel, 16, 0)
        subgrid.addWidget(self.midbeamdropdown, 17, 0)
        
        self.paramWidget.setFixedHeight(500)
        self.paramWidget.setFixedWidth(400)
        
#        subgrid.setColumnMinimumWidth(1, 130)
#        subgrid.setColumnMinimumWidth(1, 50)
        
        # file/folder widget layout
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        # EBP widget layout
        
        subgridEBP = QtGui.QGridLayout()
        self.EBPWidget.setLayout(subgridEBP)
        
        subgridEBP.addWidget(EBPparamTitle, 0, 0, 2, 4)
        
        subgridEBP.addWidget(self.EBProiButton, 2, 0, 1, 1)
        subgridEBP.addWidget(self.set_EBPButton, 3, 0, 1, 1)
        subgridEBP.addWidget(self.showEBPButton, 4, 0, 2, 1)
        subgridEBP.addWidget(self.EBPtypeLabel, 2, 1)
        subgridEBP.addWidget(self.EBPtype, 3, 1)
        subgridEBP.addWidget(self.Llabel, 4, 1)
        subgridEBP.addWidget(self.LEdit, 5, 1)
        
        self.EBPWidget.setFixedHeight(200)
        self.EBPWidget.setFixedWidth(300)
        
        # Piezo navigation widget layout

        layout = QtGui.QGridLayout()
        self.positioner.setLayout(layout)
        
        positionerTitle = QtGui.QLabel('<h2><strong>Position</strong></h2>')
        positionerTitle.setTextFormat(QtCore.Qt.RichText)
        
        layout.addWidget(positionerTitle, 0, 0, 2, 3)
        layout.addWidget(self.xUpButton, 2, 4, 2, 1)
        layout.addWidget(self.xDownButton, 2, 2, 2, 1)
        
        layout.addWidget(self.xStepLabel, 0, 6)        
        layout.addWidget(self.xStepEdit, 1, 6)
        
        layout.addWidget(self.yUpButton, 1, 3, 2, 1)
        layout.addWidget(self.yDownButton, 3, 3, 2, 1)
        
        layout.addWidget(self.yStepLabel, 2, 6)        
        layout.addWidget(self.yStepEdit, 3, 6)

        layout.addWidget(self.zUpButton, 1, 5, 2, 1)
        layout.addWidget(self.zDownButton, 3, 5, 2, 1)
        
        layout.addWidget(self.zStepLabel, 4, 6)        
        layout.addWidget(self.zStepEdit, 5, 6)
        
        layout.addWidget(self.moveToLabel, 6, 1, 1, 3)
        layout.addWidget(self.moveToEdit, 7, 1, 1, 2)
        layout.addWidget(self.moveToButton, 8, 1, 1, 2)
        
        self.positioner.setFixedHeight(250)
        self.positioner.setFixedWidth(400)
        
        
        
        intensityTitle = QtGui.QLabel('<h2><strong>Intensity settings</strong></h2>')
        intensityTitle.setTextFormat(QtCore.Qt.RichText)      
        
        self.intensitySlider = QSlider(Qt.Horizontal)
        self.intensitySlider.setGeometry(30, 40, 200, 30)
#        self.intensitySlider.setRange(0, 4)
        self.intensitySlider.setMinimum(0.04)
        self.intensitySlider.setMaximum(40)
        self.intensitySlider.setTickInterval(5)
        self.intensitySlider.setSingleStep(1)
        self.intensitySlider.setTickPosition(QSlider.TicksBothSides)
        self.intensitySlider.setValue(30)
        
        
        layout = QtGui.QGridLayout()
        self.intensityWidget.setLayout(layout)
        

        layout.addWidget(intensityTitle, 0,0)
        layout.addWidget(self.intensitySlider, 1,0)
        layout.addWidget(self.intensityLabel, 2,0)
        layout.addWidget(self.intensityEdit, 2,1)
        
        layout.addWidget(self.offsetLabel, 3,0)
        layout.addWidget(self.offsetEdit, 3,1)
    # make connections between GUI and worker functions
            
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)
        backend.imageSignal.connect(self.get_image)
        backend.finishmeasurementSignal.connect(self.releaseLiveviewButton)
        backend.toggleShutterSignal.connect(self.toggleShutterButton)
        backend.found_fit_Signal.connect(self.center_Crosshair)
        
    def closeEvent(self, *args, **kwargs):

        # Emit close signal

        self.closeSignal.emit()
        
        
      
class Backend(QtCore.QObject):
    
    paramSignal = pyqtSignal(dict)
    imageSignal = pyqtSignal(np.ndarray)
    imageSignalPD = pyqtSignal(np.ndarray)
    frameIsDone = pyqtSignal(bool, np.ndarray) 
    ROIcenterSignal = pyqtSignal(np.ndarray)
    tcspcPrepareScanSignal = pyqtSignal(int)
    tcspcPrepareHistoScanSignal = pyqtSignal(int)
    tcspcMeasureScanSignal = pyqtSignal()
    tcspcMeasureFastScanSignal = pyqtSignal()
    tcspcMeasureScanPointSignal = pyqtSignal()
    tcspcCloseConnection = pyqtSignal()
    tcspcrequestHHStatusSignal = pyqtSignal()
    finishmeasurementSignal =  pyqtSignal()
    toggleShutterSignal = pyqtSignal(bool)
    
    found_pick_PickandDestroy = pyqtSignal(bool)
    found_fit_Signal = pyqtSignal(np.ndarray)
    
    setAOTFonSignal = pyqtSignal(bool)
    setgreenAOTFonSignal = pyqtSignal(bool)
    setredAOTFonSignal = pyqtSignal(bool)
    
    
    """
    Signals
    
    - paramSignal:
         To: [frontend]
         Description: 
             
    - imageSignal:
         To: [frontend]
         Description: 

    - imageSignalPD:
         To: [frontend]
         Description: 
             
             
    - finishmeasurementSignal
        To:  [frontend]
        Description: Release liveview button 
        
    - toggleShutterSignal
        To:  [frontend]
        Description: Toggles Shutter Button
                     
        
    - frameIsDone:
         To: [psf]
         Description: 
        
    - ROIcenterSignal:
         To: [minflux]
         Description:
                         
             
    - tcspcPrepareSignal:
         To: [tcspc]
         Description:
             
             
    - tcspcrequestHHStatusSignal
    
         To: [tcspc]
         Description:    
    
    - tcspcMeasureScanSignal:
         To: [tcspc]
         Description:
             
    - found_pick_PickandDestroy:

        To: [PickAndDestroy]
        Description:  found Particle -> measure MINFLUX
             
    - found_fit_Signal:
        
        To: [frontend]
        Description: center it
        
        
    - setAOTFonSignal:
        
        To: [aotf]
        Description: sets both AOTF colors on/off
        
        
        
    - setredAOTFonSignal:
        
        To: [aotf]
        Description: sets only red AOTF on/off
        
    - setgreenAOTFonSignal:
        
        To: [aotf]
        Description: sets only green AOTF on/off
        
    """
    
    def __init__(self, pidevice, owisps10, scanTCSPCdata, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.pidevice = pidevice
        self.saveScanData = False
        self.feedback_active = False
        self.scanTCSPCdata = scanTCSPCdata # queue for synchonisation of SCAN + TCSPC measurement 
        self.linedone = True # Flag that no line scan is in progress
        #if not standAlone:
        self.standAlone = False
        self.shutter_state = False
        
        self.owisps10 = owisps10

        self.initflag = True
        
        # 5MHz is max count rate of the P. Elmer APD
        self.APDmaxCounts = 1*10**6   
        self.fast_scan = True #TODO: does it need to be removed?
        self.ODslope = 0.0148
        
        
        # Create a timer for the update of the liveview
        self.angle = 0
        self.viewtimer = QtCore.QTimer()
        self.measuretimer = QtCore.QTimer()
        # Counter for the saved images

        self.imageNumber = 0
        self.frameTime = 0
        # initialize flag for the linescan function

        self.flag = 0
        

        # initial directory

        self.initialDir = r'C:\Data'
        self.ODconfigdir = r'C:\Users\MINFLUX\PYFLUX\Development\PYFLUX\config\\'
        self.ODconfigname = os.path.join(self.ODconfigdir, "ODconfig")
        tools.readODconfig(self,self.ODconfigname)
        
        #Reinit OD angle
        initpos = self.owisps10.getPosition(1) 
        print("Current OD filter position: ",initpos)
#        if (initpos < 1) & (initpos > 1):
        print("Recalibration of ND filter from file")
        
#            self.ODangleoffsetinit
#            self.ODinit
#            self.angle
#            
        self.newoffset = float(self.ODinit) / float(self.ODslopeinit) - initpos
        print(self.newoffset)
        # initialize image
        
        self.image = None
        
        
        
        #NIDAQ
        self.BitTask1 = Task()
        self.BitTask2 = Task()
        self.BitTask3 = Task()
        self.BitTask4 = Task()
        self.BitTask5 = Task()
        self.BitTask6 = Task()
        self.BitTask7 = Task()


        
        self.BitTask1.do_channels.add_do_chan("Dev1/port1/line3") # this is digital pin0 of port1 on Device 1
        self.BitTask2.do_channels.add_do_chan("Dev1/port1/line2") # this is digital pin1 of port1 on Device 1
        self.BitTask3.do_channels.add_do_chan("Dev1/port1/line1") # this is digital pin1 of port1 on Device 1
        
        #red
        self.BitTask4.do_channels.add_do_chan("Dev1/port1/line0") # this is digital pin1 of port1 on Device 1
        self.BitTask5.do_channels.add_do_chan("Dev1/port0/line7") # this is digital pin1 of port1 on Device 1
        self.BitTask6.do_channels.add_do_chan("Dev1/port0/line6") # this is digital pin1 of port1 on Device 1
        self.BitTask7.do_channels.add_do_chan("Dev1/port0/line5") # this is digital pin1 of port1 on Device 1


        #Shutter Bits
        self.List_Bit   = [self.BitTask1,self.BitTask2, self.BitTask3, self.BitTask4, self.BitTask5, self.BitTask6, self.BitTask7]
        
        time.sleep(0.2)
#        self.toggle_shutter_all(False)
        
        #toggle shutters
        self.SetBitOff(self.BitTask1)
        self.SetBitOff(self.BitTask2)
        self.SetBitOff(self.BitTask3)
        self.SetBitOff(self.BitTask4)
        self.SetBitOff(self.BitTask5)
        self.SetBitOff(self.BitTask6)
        self.SetBitOff(self.BitTask7)
        
        
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        # updates parameters according to what is input in the GUI
        
        self.scandirection = params['scanDirectionType']
        self.scancolor = params['scancolor']
        
        if self.scancolor == 'green/red':
            self.dualcolorflag = True
        else:
            self.dualcolorflag = False
        
        self.scantype = params['scanType']
        self.scanRange = params['scanRange'] 
        
        self.NofPixels = int(params['NofPixels'])
        self.pxTime = params['pxTime']
        
        self.initialPos = params['initialPos']
        
        self.waitingTime = params['waitingTime']
        self.pulse_bin = params['pulse_bin']
        
        self.filename = params['fileName']
        
        self.moveToPos = params['moveToPos']
        
        self.xStep = params['xStep']
        self.yStep = params['yStep']
        self.zStep = params['zStep']
        self.OD = params['intensitySlider'] /10
        
        self.lifetimelimit = params['lifetimelimit']

        self.ODangleoffset = params['intensityOffset']
        
        self.midbeam = params['midbeam']
        
        self.lifetimeflag = params['lifetimeflag']
        
        
                
        self.scanOD = params['scanOD']
        self.tcspcOD = params['tcspcOD']




#        print("lfflag ", self.lifetimeflag)
        
        self.calculate_derived_param()
        
    def calculate_derived_param(self):
        
        self.image_to_save = self.image
        
        #Thorlabs
        #OD = mθ , m = 0.0148
        
        
        print( self.ODangleoffset)
        
        self.angle = self.OD/ self.ODslope - self.ODangleoffset
        print(self.angle)
        now = time.strftime("%c")
        tools.saveODconfig(self, now, self.ODconfigname )
        print("save")

        self.owisps10.moveAbsolute(1, int(self.angle))
        
        
        self.pxSize = self.scanRange/self.NofPixels   # in µmhttps://webmail.cup.uni-muenchen.de/webmail/rcube/?_task=mail&_action=get&_mbox=INBOX&_uid=7278&_token=jWhh9Lq37s3PFiYWdXtIvUSQ736SKiQH&_part=2
        self.velocity = self.pxSize / self.pxTime * 10**6 #vel in µm/s
        print("self.velc", self.velocity)
        
        self.borderadd = 1.10
        self.imageincrease = self.borderadd * 2 - 1
        
        
        
        #TODO: only during scan
        self.pidevice.VEL("2", 10000)
        self.pidevice.VEL("3", 5000)
        if self.fast_scan:
            if self.scantype == 'xy':
                self.pidevice.VEL("2", self.velocity)
            if self.scantype == 'xz' or 'yz':
                self.pidevice.VEL("3", self.velocity)
        elif self.fast_scan == False:
            self.pidevice.VEL("2", 10000)
            
            
        
        
        
        self.frameTime = self.NofPixels**2 * self.pxTime / 10**6
        print(self.frameTime)
        self.maxCounts = int(self.APDmaxCounts/(1/(self.pxTime*10**-6)))
        self.linetime = (1/1000)*self.pxTime*self.NofPixels  # in ms
        

        # create scan signal

        self.dy = self.pxSize
        self.dz = self.pxSize
        #Zeiten bei denen er bei Positionen  sein soll?
        
        #TODO: remove data_t?
        self.data_t = np.arange(0, self.pxTime * self.NofPixels, self.pxTime )
        
        #TODO: make checks to not drive > max Range
        
        if self.scantype == 'xy':
            self.data_x = np.arange(self.initialPos[0] , self.initialPos[0] + self.scanRange ,\
                                self.scanRange / self.NofPixels)
            self.data_y = np.arange(self.initialPos[1], self.initialPos[1] + self.scanRange ,\
                                self.scanRange  / self.NofPixels) 
            
        elif self.scantype == 'xz' or self.scantype == 'yz':  
            

            self.data_x = np.arange(self.initialPos[0] , self.initialPos[0]+ self.scanRange,\
                                self.scanRange / self.NofPixels )
            self.data_y = np.arange(self.initialPos[1] , self.initialPos[1] + self.scanRange,\
                                self.scanRange / self.NofPixels   ) 
                        
            self.data_z = np.arange(self.initialPos[2]- self.scanRange/2 , self.initialPos[2] + self.scanRange/2,\
                                    self.scanRange / self.NofPixels) 

        
                          

        
        self.viewtimer_time = 0  # timer will timeout as soon after it has executed all functions

        size = (self.NofPixels, self.NofPixels)
        sizelf = (self.NofPixels, self.NofPixels, 4)
        sizelf2 = (self.NofPixels, self.NofPixels, 3)


        self.blankImage = np.zeros(size)
        self.image = self.blankImage
        self.imagered = np.copy(self.blankImage)
        self.imagegreen = np.copy(self.blankImage)
        
        
        self.lifetimearray = np.zeros(size)
        self.lifetimeimage = np.zeros(sizelf2)
        self.dualcolorimage = np.zeros(sizelf)
        self.i = 0

        
        self.update_device_param()

        # emit calculated parameters

        self.emit_param()
        
    def emit_param(self):
        
        params = dict()
        
        
        
        params['ODinit'] = self.ODinit
        params['ODangleoffsetinit'] = self.ODangleoffsetinit
        
        params['frameTime'] = self.frameTime
        params['pxSize'] = self.pxSize
        params['maxCounts'] = self.maxCounts
        params['initialPos'] = np.float64(self.initialPos)
        
        self.paramSignal.emit(params)
        
    def update_device_param(self):
        #TODO opens at pressed button! merge with calculate?

        #pass

        #initial positions x and y
        
        self.x_i = self.initialPos[0]
        self.y_i = self.initialPos[1]
        self.z_i = self.initialPos[2]
        
        
        if self.initflag:

            self.numberx = 0
            self.numbery = 0
            self.initflag = False #TODO:reset at liveview ?true?
            self.calculate_derived_param() 
            self.t000 = time.time()




    def set_moveTo_param(self, x_f, y_f, z_f ):
        self.pidevice.MOV('1', x_f)
        self.pidevice.MOV('2', y_f)
        self.pidevice.MOV('3', z_f)
        
        time.sleep(0.05)
        for axis in self.pidevice.axes:
            position = self.pidevice.qPOS(axis)[axis]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
            print('current position of axis {} is {:.2f}'.format(axis, position))


    def set_moveToXY(self, x_f, y_f, t = 0.01):
        self.pidevice.MOV('1', x_f)
        self.pidevice.MOV('2', y_f)
        time.sleep(t)#TODO:fix no fixed time, but adapted ... wait also sleep?
        
    def set_moveToXZ(self, x_f, z_f, t = 0.01):
        self.pidevice.MOV('1', x_f)
        self.pidevice.MOV('3', z_f)
        time.sleep(t)
        
    def set_moveToYZ(self, y_f, z_f, t = 0.01):
        self.pidevice.MOV('2', y_f)
        self.pidevice.MOV('3', z_f)
        time.sleep(t)
        
        


    def moveTo(self, x_f, y_f, z_f):

        self.set_moveTo_param(x_f, y_f, z_f)

    def moveTo_action(self):

        self.moveTo(*self.moveToPos)
        
    def moveTo_roi_center(self):
        
        self.ROIcenter = self.initialPos + np.array([self.scanRange/2, self.scanRange/2, 0])
        
        print('[scan] self.initialPos[0:2]', self.initialPos[0:2])
        print('[scan] center', self.ROIcenter)
        
        self.moveTo(*self.ROIcenter)
        self.ROIcenterSignal.emit(self.ROIcenter)
        
#        # keep track of the new position to where you've moved
#        
#        self.initialPos = self.ROIcenter
#        self.emit_param()
        
    @pyqtSlot()
    def get_moveTo_initial_signal(self):
        
        self.moveTo(*self.initialPos)
    
    def relative_move(self, axis, direction):
        
        current_position = np.zeros(3)
        
        for piaxis in self.pidevice.axes:
            current_position[int(piaxis) - 1] = self.pidevice.qPOS(piaxis)[piaxis]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
        
        print(current_position)
        
        if axis == 'x' and direction == 'up':
            
            newPos_µm = current_position[0] + self.xStep
            newPos_µm = round(newPos_µm, 3)
            position = np.array([newPos_µm, current_position[1],
                                        current_position[2]])
            
        if axis == 'x' and direction == 'down':
            
            newPos_µm = current_position[0] - self.xStep
            newPos_µm = np.around(newPos_µm, 3)
            position = np.array([newPos_µm, current_position[1],
                                        current_position[2]])
            
        if axis == 'y' and direction == 'up':
            
            newPos_µm = current_position[1] + self.yStep
            newPos_µm = np.around(newPos_µm, 3)       
            position = np.array([current_position[0], newPos_µm,
                                        current_position[2]])
            
        if axis == 'y' and direction == 'down':
            
            newPos_µm = current_position[1] - self.yStep
            newPos_µm = np.around(newPos_µm, 3)
            position = np.array([current_position[0], newPos_µm,
                                        current_position[2]])
            
        if axis == 'z' and direction == 'up':
            
            newPos_µm = current_position[2] + self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            position = np.array([current_position[0], current_position[1], 
                                        newPos_µm])
        
        if axis == 'z' and direction == 'down':
            
            newPos_µm = current_position[2] - self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            position = np.array([current_position[0], current_position[1], 
                                        newPos_µm])
        print(position)
        self.update_device_param()
        self.moveTo(position[0], position[1], position[2])
        self.emit_param()    
            
    def save_current_frame(self):
        
        # experiment parameters
        
        name = tools.getUniqueName(self.filename)
        print(name)
        now = time.strftime("%c")
        tools.saveConfig(self, now, name)
        
        # save image
        
        data = self.image_to_save
        print(data)
        result = Image.fromarray(data.astype('uint16'))
        #added rotate 270 check
        result.rotate(270).save(r'{}.tiff'.format(name),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
        
        
        relTimeslistarray = np.concatenate(self.relphotons).ravel() # in (ns)
        np.savetxt(name+ 'relphotons.txt', relTimeslistarray)
#        np.savetxt(name + "absphotons.txt", self.absphotons)
        np.savetxt(name + "image.txt", self.image.astype(int), fmt='%i') #TODO: check if fmt i does problems with 10
#         

#        np.savetxt(r'{}_relphotons.txt'.format(name), self.relphotons)
#        np.savetxt(r'{}_absphotons.txt'.format(name), self.absphotons)
#        np.savetxt(r'{}_imagedata.txt'.format(name), self.image)
#        
        
        print('[scan] Saved current frame', name)





    def save_current_TCSPC(self):
        
        
        name = tools.getUniqueName(self.filename+"_relTimes")
        

        relTimeslistarray = np.concatenate(self.relphotons).ravel() * 1e9  # in (ns)
        
        np.savetxt(name, relTimeslistarray.T)

        print('[scan] Saved current Frames TCSPC data')
        

    @pyqtSlot(bool, str, np.ndarray, str)
    def get_scan_signal(self, lvbool, mode, initialPos, filename):
    
        """
        Connection: [psf] scanSignal
        Description: get drift-corrected initial position, calculates the
        derived parameters 
        """
        self.psffilename = filename
        #self.initialPos = initialPos
        self.calculate_derived_param()
        
        self.liveview(lvbool, mode, timegate = True)

        
  


    def fastline_scan(self, timearr, xpos, ypos ): #xy -> -yx
        """
        Connection: TO [tcspc]  
        """ 
        self.linedone = False
         # get min max and current position
        #rangemin = pidevice.qTMN()
        #rangemax = pidevice.qTMX()
        TCSPCdone = False
        
        self.set_moveToXY(xpos , ypos[0] - self.scanRange * (self.borderadd -1), t = 0.05)
        
#        time
        line_data = np.zeros(self.NofPixels)
        
     
        self.set_moveToXY(xpos , ypos[-1] + self.scanRange * (self.borderadd -1), t = 0) 
        
#        time.sleep(0.006)
        self.tcspcMeasureFastScanSignal.emit() 
        
                
        pitools.waitontarget(self.pidevice, axes='1', polldelay = 0.0) 
        
        while TCSPCdone == False:
            try:
                TCSPCdone, count, reltimes, channel_array = self.scanTCSPCdata.get(timeout=0)
            except queue.Empty:
                pass
        
        
        self.scanTCSPCdata.queue.clear()
        offset = -5
        #Bin abs photon times into into pixels
        line_data = np.histogram(count, bins = int(self.NofPixels), range =  (self.pxTime * (self.NofPixels * (self.borderadd -1) + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1) )* 10**-3, self.pxTime * (self.NofPixels * self.borderadd + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1))* 10**-3)) #count in ms
        
        colorcode = np.zeros(self.NofPixels)
            
        self.linedone = True 
#        print(line_data)
        return line_data[0], colorcode, count, reltimes, channel_array
    
    def fastline_scan_xz(self, timearr, xpos, ypos ): #xy -> -yx
        """
        Connection: TO [tcspc]  
        """ 
        self.linedone = False
        TCSPCdone = False
        
        self.set_moveToXZ(xpos , ypos[0] - self.scanRange * (self.borderadd -1), t = 0.05)
        
        
        line_data = np.zeros(self.NofPixels)

        self.set_moveToXZ(xpos , ypos[-1] + self.scanRange * (self.borderadd -1), t = 0) 

        self.tcspcMeasureFastScanSignal.emit() 
        
        
        
        pitools.waitontarget(self.pidevice, axes='2', polldelay = 0.0)

#        t01 = time.time()
        
            
        
        while TCSPCdone == False:
            try:
                TCSPCdone, count, reltimes, channel_array = self.scanTCSPCdata.get(timeout=0)
            except queue.Empty:
                pass
               
        self.scanTCSPCdata.queue.clear()
###old
#        offset = -5
#        #Bin abs photon times into into pixels
#        line_data = np.histogram(count, bins = int(self.NofPixels), range =  (self.pxTime * (self.NofPixels * (self.borderadd -1) + offset *self.pxTime/500 * (self.NofPixels/100) )* 10**-3, self.pxTime * (self.NofPixels * self.borderadd + offset *self.pxTime/500* (self.NofPixels/100))* 10**-3)) #count in ms
#        
        offset = -5
        #Bin abs photon times into into pixels
        line_data = np.histogram(count, bins = int(self.NofPixels), range =  (self.pxTime * (self.NofPixels * (self.borderadd -1) + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1) )* 10**-3, self.pxTime * (self.NofPixels * self.borderadd + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1))* 10**-3)) #count in ms
        
        colorcode = np.zeros(self.NofPixels)
            
        self.linedone = True 
        return line_data[0], colorcode, count, reltimes, channel_array
    
    def fastline_scan_allaxis(self, timearr, slowaxis, fastaxis, scantype): #xy -> -yx
        """
        Connection: TO [tcspc]  
        """ 
        #TODO: check if needed probably single movements better
        self.linedone = False
         # get min max and current position
        #rangemin = pidevice.qTMN()
        #rangemax = pidevice.qTMX()
        TCSPCdone = False
        if scantype == 'xy':
            self.set_moveToXY(slowaxis , fastaxis[0], t = 0.05)
        
        if scantype == 'xz':
            self.set_moveToXZ(slowaxis , fastaxis[0], t = 0.05)
        
        if scantype == 'yz':
            self.set_moveToYZ(slowaxis , fastaxis[0], t = 0.05)
            
            
            
        line_data = np.zeros(self.NofPixels)
        
        t0 = time.time()
        #print(datetime.now(), '[scan] The measurement started ')
        self.tcspcMeasureFastScanSignal.emit() #TODO: check timing #increase speed for driving back?
        
        if scantype == 'xy':
            self.set_moveToXY(slowaxis , fastaxis[-1], t = 0.0)
        
        if scantype == 'xz':
            self.set_moveToXZ(slowaxis , fastaxis[-1], t = 0.0)
        
        if scantype == 'yz':
            self.set_moveToYZ(slowaxis , fastaxis[-1], t = 0.0)
        
        #self.set_moveToXY(slowaxis , fastaxis[-1], t = 0) #TODO: wwait till aligned? check time? HOW
        #pitools.moveandwait(self.pidevice, '1', xpos[-1])
        time.sleep(0.001)
        pitools.waitontarget(self.pidevice, axes='1', polldelay = 0.01) #TODO: make polldelay variable? 0.02
        
        t01 = time.time()
        
        print(datetime.now(), '[scan] The movement took {} s'.format(t01-t0))
        
        
        
        
        
        t0 = time.time()
        while TCSPCdone == False:
            try:
                TCSPCdone, count, reltimes, channel_array = self.scanTCSPCdata.get(timeout=0)
                #print("got data", TCSPCdone, count)
            except queue.Empty:
                pass
        t01 = time.time()
        print(datetime.now(), '[scan] The waiting took {} s'.format(t01-t0))
        self.scanTCSPCdata.queue.clear()
        t11 = time.time()
        line_data = np.histogram(count, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3)) #count in ms
        colorcode = np.zeros(self.NofPixels)
        
        self.linedone = True 
        
        return line_data[0][int(self.NofPixels*(self.borderadd - 1)):int(self.NofPixels*(self.borderadd))], colorcode, count, reltimes, channel_array
    
        
    @pyqtSlot(bool, str)
    def liveview(self, lvbool, mode, timegate = False):
        
        
        
        # if self.measure or self.hh.ctcdone #abfrage ob HH belegt or self
        self.numberx = 0
        self.numbery = 0
#        self.setAOTFonSignal.emit(True)
        if self.scancolor == 'green':
            self.setgreenAOTFonSignal.emit(True)
            self.setredAOTFonSignal.emit(False)
        elif self.scancolor == 'red':
            self.setredAOTFonSignal.emit(True)
            self.setgreenAOTFonSignal.emit(False)
        
        elif self.scancolor == 'green/red':
            self.setredAOTFonSignal.emit(True)
            self.setgreenAOTFonSignal.emit(True)
        
        
        
        if lvbool and timegate == False:
            self.timegate  = False
            self.acquisitionMode = mode # modes: 'liveview', 'frame'
            self.liveview_start()
        elif lvbool and timegate == True:
            self.timegate  = True
            self.acquisitionMode = mode # modes: 'liveview', 'frame'
            self.liveview_start()
        else:
            
            self.liveview_stop()

    def liveview_start(self):
        print("start scan with linetime = ", self.pxTime * self.NofPixels)

        self.measure = True 
        self.absphotons = []
        self.relphotons = []
        self.maxcount = 1 #todo change!
        self.FLIMimagechange = False
        self.setODscan()
        
        if self.scantype == 'xy':

            self.moveTo(self.x_i, self.y_i, self.z_i)

        if self.scantype == 'xz':

            self.moveTo(self.x_i, self.y_i,
                        self.z_i - self.scanRange/2)

        if self.scantype == 'yz':

            self.moveTo(self.x_i, self.y_i,
                        self.z_i - self.scanRange/2)

        self.viewtimer.start(self.viewtimer_time)
        #TODO: only at init?
        if self.fast_scan:
            self.tcspcPrepareScanSignal.emit(self.pxTime * self.NofPixels * self.imageincrease) #TODO check if prepared already
        else:
            self.tcspcPrepareHistoScanSignal.emit(self.pxTime * self.NofPixels * self.imageincrease) 
        
        
        
        time.sleep(1)
        
        if self.scancolor == 'green/red':
            self.toggle_shutter_rg(True)
            
        else:
            self.toggle_shutter(True)
        
        
        
    def liveview_stop(self):

        self.viewtimer.stop()
        
        if self.scancolor == 'green/red':
            self.toggle_shutter_rg(False)
            
        else:
            self.toggle_shutter(False)
#        self.toggle_shutter(False)
#        self.setAOTFonSignal.emit(False)


    def make_RGBA_lifetime(self):
#        print("RGBA")
        
        
# =============================================================================
#       #make RGBA image
#       
#       calculate lifetimes using ss.expon()
#       RGB = parula LUT * lifetime/maxlifetime (fixed as 3ns)
#        
#       A = count /maxcount
# =============================================================================
        
        self.maxcount = 7
        self.maxlifetime = 5 #TODOmake variable
        
        
        min_val_intensity = 5 # low value for intensity ("grey" contrast)
        max_val_intensity = 60 # high value for intensity ("grey" contrast)
        min_val_lifetime = 0.3 # low value for lifetime colorscale
        max_val_lifetime = self.lifetimelimit# high value for lifetime colorscale
            
#        min_val_intensity = np.quantile(FLIM_array_N,0.05)  
#        max_val_intensity = np.quantile(FLIM_array_N,0.99)  
        
        
        range_lifetime =  max_val_lifetime - min_val_lifetime       
        range_intensity = max_val_intensity - min_val_intensity
        
        idx_above_threshold = np.where(self.lineData> min_val_intensity)
        num_px_to_analyze = np.shape(idx_above_threshold)
        num_px_to_analyze = num_px_to_analyze[1]       
        
        
        lifetimes = np.zeros(np.shape(self.lineData))
        
#        RGBAline = []
         
        countprevious  = 0
#        lut = viewbox_tools.generatePgColormap(cmaps.parula)#
#        print(lut)
        

        for k in range(num_px_to_analyze):
            pixelnr = idx_above_threshold[0][k]
#            print(pixelnr)
#            print(idx_above_threshold)
#            print(k)
#            j = idx_above_threshold[1][k]
            
            pxcount = self.lineData[pixelnr]
#            print(pxcount)
#            print(self.lineData)

                
                
            
            if pxcount > 0:
            
                reltimes = self.relphotons[-1][countprevious: countprevious + pxcount]
            
            
                #TODO make adaptive to midbeam
                
                if self.scancolor == 'red':
                    
                    rel = reltimes[reltimes > self.pulse_bin[-1]] - self.pulse_bin[-1]
                    
                    if np.array(rel).any():
                        
                        lifetimes[pixelnr] =  ss.expon.fit(rel)[1] 
                    else:
                        pass                
                elif self.scancolor == 'green':
                    lifetimes[pixelnr] = ss.expon.fit(reltimes[reltimes < self.pulse_bin[1]]- self.pulse_bin[0])[1] 
                else:
                    lifetimes[pixelnr] = ss.expon.fit(reltimes)[1]
                
                countprevious = countprevious + pxcount
            else:
                pass
#                lifetimes.append(0)
            
            
            
            
            value_lifetime_minus_min = lifetimes[pixelnr] - min_val_lifetime      
            value_intensity_minus_min = self.lineData[pixelnr] - min_val_intensity
            
            if value_lifetime_minus_min/range_lifetime > 1:
                value_lifetime_minus_min = range_lifetime
                
            if value_lifetime_minus_min < 0:
                value_lifetime_minus_min = 0
            
            if value_intensity_minus_min/range_intensity > 1:
                value_intensity_minus_min = range_intensity
                
            if value_intensity_minus_min < 0:
                    value_intensity_minus_min = 0
            
            
            
            lifetime_ref_fraction = value_lifetime_minus_min/range_lifetime
            intensity_ref_fraction = value_intensity_minus_min/range_intensity
#            print(lifetime_ref_fraction)
        
        
            rgb_map = cm.jet(range(256)) # change map name with the map you want    
            #maps: brg
            
            rgb_map = rgb_map[:,:3]        
            color_only_lifetime = rgb_map[int((lifetime_ref_fraction*256)-1)]            
            fixed_lifetime_variable_intensity = np.zeros((256,3))
        
            fixed_lifetime_variable_intensity[:,0] = np.reshape(np.linspace(0,color_only_lifetime[0],
                                                                    num=256), (256,))
        
            fixed_lifetime_variable_intensity[:,1] = np.reshape(np.linspace(0,color_only_lifetime[1],
                                                                    num=256), (256,))
        
            fixed_lifetime_variable_intensity[:,2] = np.reshape(np.linspace(0,color_only_lifetime[2],
                                                                    num=256), (256,))     
            intensity_color = fixed_lifetime_variable_intensity[int(intensity_ref_fraction*255)]
        
#            self.lifetimeimage[self.numberx,pixelnr,:] = lifetimes
#            print(pixelnr)
            self.lifetimeimage[self.numberx,self.NofPixels -pixelnr -1,0] = intensity_color[0]
            self.lifetimeimage[self.numberx,self.NofPixels -pixelnr -1,1] = intensity_color[1]
            self.lifetimeimage[self.numberx,self.NofPixels -pixelnr -1,2] = intensity_color[2]
            
            
#        print(lifetimes)
        self.imageSignal.emit(self.lifetimeimage)
            
        return lifetimes
##            print(lifetimes)
#            color = lut.map((lifetimes[-1]) / self.maxlifetime) # todochange!!! lut is not [2,2,2]
#            color = color.astype(np.float64)/255.
##            print(pxcount)
##            print(maxcount)
#            if self.maxcount > 0:
#                
##                print("qauot", pxcount/ self.maxcount )
#                color[-1] =  pxcount/ self.maxcount #*255 # [0:1] ->  [0:255]
#            else: 
#                color[-1] = 0            
#            
#
#
#                
#            
##            print(color)
#            RGBAline.append( color)
#
#        if max(self.lineData) > self.maxcount:
#            
#            self.FLIMmaxratio = max(self.lineData)/self.maxcount
#            
#            self.maxcount = max(self.lineData)
#            self.FLIMimagechange = True
#                      
#        
#        return RGBAline
        
        
    def calculatelifetimeImage(self):
        
        min_val_intensity = 5 # low value for intensity ("grey" contrast)
        max_val_intensity = 20 # high value for intensity ("grey" contrast)
        min_val_lifetime = 0.1E-9 # low value for lifetime colorscale
        max_val_lifetime = 3.1E-9 # high value for lifetime colorscale
            
#        min_val_intensity = np.quantile(FLIM_array_N,0.05)  
#        max_val_intensity = np.quantile(FLIM_array_N,0.99)  
        
        
        range_lifetime =  max_val_lifetime - min_val_lifetime       
        range_intensity = max_val_intensity - min_val_intensity
        
        
        
        self.lifetimearray[self.numberx] 
        self.lineData
        for pixelnr in range(self.NofPixels):
            
            pxcount = self.lineData[pixelnr]
            

        
        
        idx_above_threshold = np.where(FLIM_array_N> min_val_intensity)
        num_px_to_analyze = np.shape(idx_above_threshold)
        num_px_to_analyze = num_px_to_analyze[1]        
        
        
        
        
        
        FLIM_array_final[i,j,:] = FLIM_array[i,j]
        FLIM_array_final[i,j,0] = intensity_color[0]
        FLIM_array_final[i,j,1] = intensity_color[1]
        FLIM_array_final[i,j,2] = intensity_color[2]

        
        rgb_map = cm.jet(range(256)) # change map name with the map you want    
        #maps: brg
        
        rgb_map = rgb_map[:,:3]        
        color_only_lifetime = rgb_map[int((lifetime_ref_fraction*256)-1)]            
        fixed_lifetime_variable_intensity = np.zeros((256,3))
    
        fixed_lifetime_variable_intensity[:,0] = np.reshape(np.linspace(0,color_only_lifetime[0],
                                                                num=256), (256,))
    
        fixed_lifetime_variable_intensity[:,1] = np.reshape(np.linspace(0,color_only_lifetime[1],
                                                                num=256), (256,))
    
        fixed_lifetime_variable_intensity[:,2] = np.reshape(np.linspace(0,color_only_lifetime[2],
                                                                num=256), (256,))     
        intensity_color = fixed_lifetime_variable_intensity[int(intensity_ref_fraction*255)]
        
#        return intensity_color

        
        self.lifetimeimage[self.numberx] = 1
        self.imageSignal.emit(self.lifetimeimage)


    def update_view(self):
        
        
        pitools.waitontarget(self.pidevice, polldelay = 0.01)
        if self.scantype == 'xy':

            if self.linedone and self.numberx < self.NofPixels: 
                if self.numberx % 2 == 1 and self.scandirection == 'bidirectional': # odd going back 
                    ylist = self.data_y[::-1]
                elif self.numberx % 2 == 0 and self.scandirection == 'bidirectional': # even 
                    ylist = self.data_y
                elif self.scandirection == 'monodirectional':
                    ylist = self.data_y[::-1]
                

                if self.fast_scan:
                    [self.lineData, colordata, absphoton, relphoton, channel_array] = self.fastline_scan(self.data_t,self.data_x[self.numberx], ylist  )
                    self.absphotons.append(absphoton)
                    self.relphotons.append(relphoton)
                else: 
                    self.lineData = self.line_scan(self.data_t,self.data_x[self.numberx], ylist  )
                if self.numberx % 2 == 0 and self.scandirection == 'bidirectional':
                    self.image[self.numberx] = self.lineData
                else:

                    self.image[self.numberx] = np.flip(self.lineData)
                
                if self.scandirection == 'monodirectional': #Going back doppelt?
                    self.set_moveToXY(self.data_x[self.numberx], ylist[0], t = 0.01)
                
                self.image_to_save = self.image
                
                if self.lifetimeflag:
                    
                    
                    
                    
                    
                    if self.FLIMimagechange:
                        
                        
                        self.lifetimearray[self.numberx] = np.flip(self.make_RGBA_lifetime()) #* self.FLIMmaxratio
                        
#                        self.lifetimearray *= self.FLIMmaxratio #self.lifetimearray.max()
                        
                        self.FLIMimagechange = False
                    else:
                        self.lifetimearray[self.numberx] = np.flip(self.make_RGBA_lifetime())
                    
                    
#                    self.calculatelifetimeImage()
                elif self.dualcolorflag == True:
                    self.calculate_dualcolor_image(absphoton, channel_array)
                    self.imageSignal.emit(self.dualcolorimage)
                    
                    
                else:
                    self.imageSignal.emit(self.image)
                
                
                
                
                
                
#                print(datetime.now(), '[scan] Image emitted to frontend')
    
                self.numberx = self.numberx + 1
                
            elif self.linedone and self.numberx == len(self.data_x) and self.measure:
                self.toggle_shutter(False)
                self.finishmeasurementSignal.emit()
                self.imageSignalPD.emit(self.image)
#                self.tcspcCloseConnection.emit() 
    
                if self.acquisitionMode == 'frame':
                    
                    self.liveview_stop()
                    
                    
                    self.frameIsDone.emit(True, self.image) #TODO: check if needed to change
    
                self.update_device_param()  
                self.measure = False
                
                
            if self.timegate:
                self.timegate_image()               
                
                
        if self.scantype == 'xz':
            if self.linedone and self.numberx < self.NofPixels: 
                if self.numberx % 2 == 1 and self.scandirection == 'bidirectional': # odd going back 
                    zlist = self.data_z[::-1]
                elif self.numberx % 2 == 0 and self.scandirection == 'bidirectional': # even 
                    zlist = self.data_z
                elif self.scandirection == 'monodirectional':
                    zlist = self.data_z[::-1]
                #print("ypos ", self.data_y[self.numberx])
                if self.fast_scan:
                    #self.lineData = self.fastline_scan_allaxis(self.data_t,self.data_x[self.numberx], zlist, "xz"  )
                     [self.lineData, colordata, absphoton, relphoton, channel_array] = self.fastline_scan_xz(self.data_t,self.data_x[self.numberx], zlist  )
                     self.absphotons.append(absphoton)
                     self.relphotons.append(relphoton)
                
                if self.numberx % 2 == 1 and self.scandirection == 'bidirectional':
                    self.image[self.numberx] = self.lineData
                else:
                    self.image[self.numberx] = np.flip(self.lineData)
                
                if self.numberx % 2 == 1 or self.scandirection == 'monodirectional': #Going back
                    self.set_moveToXZ(self.data_x[self.numberx], zlist[1], t = 0)
                
                self.image_to_save = self.image
                
                
                
                if self.dualcolorflag == True:
                    self.calculate_dualcolor_image(absphoton, channel_array)
                    self.imageSignal.emit(self.dualcolorimage)
                
                else:
                    self.imageSignal.emit(self.image)
                print(datetime.now(), '[scan] Image emitted to frontend')
    
                self.numberx = self.numberx + 1
                
            elif self.linedone and self.numberx == self.NofPixels and self.measure:
                self.toggle_shutter(False)
                self.finishmeasurementSignal.emit()
                self.tcspcCloseConnection.emit() 
                
                if self.acquisitionMode == 'frame':
                    
                    self.liveview_stop()
                    self.frameIsDone.emit(True, self.image)
    
                self.update_device_param()  
                self.measure = False
            
        
        
        if self.scantype == 'yz':

            if self.linedone and self.numbery < self.NofPixels: 
                if self.numbery % 2 == 1 and self.scandirection == 'bidirectional': # odd going back 
                    zlist = self.data_z[::-1]
                elif self.numbery % 2 == 0 and self.scandirection == 'bidirectional': # even 
                    zlist = self.data_z
                elif self.scandirection == 'monodirectional':
                    zlist = self.data_z[::-1]
                #print("ypos ", self.data_y[self.numberx])
                if self.fast_scan:
                    [self.lineData, colordata, self.absphotons[self.numbery], self.relphotons[self.numbery], channel_array ] = self.fastline_scan_allaxis(self.data_t,self.data_y[self.numbery], zlist, "yz"  )
                
                if self.numberx % 2 == 1 and self.scandirection == 'bidirectional':
                    self.image[self.numbery] = self.lineData
                else:
                    self.image[self.numbery] = np.flip(self.lineData)
                
                if self.numberx % 2 == 1 or self.scandirection == 'monodirectional': #Going back
                    self.set_moveToXZ(self.data_y[self.numbery], zlist[0], t = 0)
                
                self.image_to_save = self.image
                self.imageSignal.emit(self.image)
                print(datetime.now(), '[scan] Image emitted to frontend')
    
                self.numbery = self.numbery + 1
                
            elif self.linedone and self.numbery == len(self.data_y) and self.measure:
                
                self.toggle_shutter(False)
                
                self.finishmeasurementSignal.emit()
                self.tcspcCloseConnection.emit() 
    
                if self.acquisitionMode == 'frame':
                    
                    self.liveview_stop()
                    self.frameIsDone.emit(True, self.image)
    
                self.update_device_param()  
                self.measure = False
                
                
                if self.timegate:
                    self.timegate_image()

    def calculate_dualcolor_image(self, absphoton, channel_array):
        
        offset = -5
        
        #make red image
        countred = absphoton[channel_array == 1]
        
        
        line_data_red = np.histogram(countred, bins = int(self.NofPixels), range =  (self.pxTime * (self.NofPixels * (self.borderadd -1) + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1) )* 10**-3, self.pxTime * (self.NofPixels * self.borderadd + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1))* 10**-3)) #count in ms
#        print(self.NofPixels)
#        print(line_data_red)
#        print(self.imagered.shape)
        
        self.imagered[self.numberx] = np.flip(line_data_red[0])
        
        
        
        #make green image
        countgreen = absphoton[channel_array == 0]
        line_data_green = np.histogram(countgreen, bins = int(self.NofPixels), range =  (self.pxTime * (self.NofPixels * (self.borderadd -1) + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1) )* 10**-3, self.pxTime * (self.NofPixels * self.borderadd + offset *self.pxTime/500 + ((110 / self.NofPixels )*3.4-1))* 10**-3)) #count in ms
        self.imagegreen[self.numberx] = np.flip(line_data_green[0])
        
        
#        H0, xedg, yedg = np.histogram2d(locs[:,0,0], locs[:,0,1], bins = binning2)
#        H1, xedg2, yedg2 = np.histogram2d(locs[:,1,0], locs[:,1,1], bins = binning2)
#        fontprops = fm.FontProperties(size=1)
#        fig_2 = plt.figure()
#        ax_2 = fig_2.add_subplot()
        Int = self.imagegreen + self.imagered
        Col = self.imagered - self.imagegreen
        Col[Int!=0] = Col[Int!=0]/(Int[Int!=0])
        Col = (Col+1)/2
        Int = Int/np.max(Int)
        self.dualcolorimage = np.stack((1-Col, np.zeros(Col.shape), Col, Int), axis = -1)
#        display = ax_2.imshow(im_array, extent = extent2)
#        scalebar = AnchoredSizeBar(ax_2.transData,
#                        10, label = None, loc =  'lower right',
#                        borderpad=13,
#                        color='black',
#                        frameon=False,
#                        size_vertical=1,
#                        fontproperties=fontprops) 
#        
#        
#        Int = H0+H1
#        Col = H0-H1
#        Col[Int!=0] = Col[Int!=0]/(Int[Int!=0])
#        Col = (Col+1)/2
##        Int = Int/np.max(Int)
#        
#        
#        
#        self.dualcolorimage[self.numberx] = np.flip(self.lineData)
        
        
        
        
    def timegate_image(self):
        """
        Calculate 4 timegated PSF out of the scan
        
        
        """
        #self.absphotons[self.numberx], self.relphotons[self.numberx] 
        print("into timegate")
        #line_data = np.histogram(count, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3)) #count in ms
        reltimes = self.pulse_bin #[0.1, 12.6, 25.2, 37.3] # TODO: change check ns
        """PSF200 = np.zeros(len(self.absphotons))
        PSF455 = np.zeros(len(self.absphotons))
        PSF710 = np.zeros(len(self.absphotons))
        PSF965 = np.zeros(len(self.absphotons))
        """
        
        size = (self.NofPixels, self.NofPixels)

        PSF200 = np.zeros(size)
        PSF455 = np.zeros(size)
        PSF710 = np.zeros(size)
        PSF965 = np.zeros(size)
        
        PSF200_rel = np.zeros(len(self.relphotons))
        PSF455_rel = np.zeros(len(self.relphotons))
        PSF710_rel = np.zeros(len(self.relphotons))
        PSF965_rel = np.zeros(len(self.relphotons))
        
        
        for i in range(len(self.absphotons)): #per line
            
            #which photons are in bigger than the bin
            photons0 = self.relphotons[i] > reltimes[0] 
            photons1 = self.relphotons[i] > reltimes[1] 
            photons2 = self.relphotons[i] > reltimes[2] 
            photons3 = self.relphotons[i] > reltimes[3] 
            
            
            #which photons are in reltimes bin
            photons200 = photons0 * np.logical_not(photons1)
            photons455 = photons1 * np.logical_not(photons2)
            photons710 = photons2 * np.logical_not(photons3)
            photons965 = photons3
            
            #get photons:
            PSF200_zeros = self.absphotons[i][photons200]
            PSF455_zeros = self.absphotons[i][photons455]
            PSF710_zeros = self.absphotons[i][photons710]
            PSF965_zeros = self.absphotons[i][photons965]
            
            #delete zeros
            PSF200_nonzero = PSF200_zeros[PSF200_zeros != 0]
            PSF455_nonzero = PSF455_zeros[PSF455_zeros != 0]
            PSF710_nonzero = PSF710_zeros[PSF710_zeros != 0]
            PSF965_nonzero = PSF965_zeros[PSF965_zeros != 0]
#            print("start ", PSF200_nonzero)
#            print(PSF710_nonzero)
            #print(PSF200_nonzero.shape)
            #print(np.flip(np.histogram(PSF200_nonzero, bins = self.NofPixels-1, range = (0, self.pxTime * self.NofPixels * 10**-3)))[0].shape)
            #BIN it and flip it according to liveview.
            PSF200[i] = np.flip(np.histogram(PSF200_nonzero, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3))[0]) #count in ms
            PSF455[i] = np.flip(np.histogram(PSF455_nonzero, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3))[0]) #count in ms
            PSF710[i] = np.flip(np.histogram(PSF710_nonzero, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3))[0]) #count in ms
            PSF965[i] = np.flip(np.histogram(PSF965_nonzero, bins = self.NofPixels, range = (0, self.pxTime * self.NofPixels * 10**-3))[0]) #count in ms
#            print("start ", PSF200)
#            print(PSF710)
        
        # experiment parameters
        
        name = tools.getUniqueName("PSF_config")
        name200 = tools.getUniqueName(self.psffilename+"_200")
        name455 = tools.getUniqueName(self.psffilename+"_455")
        name710 = tools.getUniqueName(self.psffilename+"_710")
        name965 = tools.getUniqueName(self.psffilename+"_965")
        
        now = time.strftime("%c")
        #tools.saveConfig(self, now, name)

#        print("start ", PSF200)
#        print(PSF710)
        #print(PSF200)
        #print(type(PSF200))
        result_200 = Image.fromarray(PSF200.astype('uint16'))
        result_455 = Image.fromarray(PSF455.astype('uint16'))
        result_710 = Image.fromarray(PSF710.astype('uint16'))
        result_965 = Image.fromarray(PSF965.astype('uint16'))
#
#        print("start ", result_200)
#        print(result_710)        

        result_200.save(r'{}.tiff'.format(name200),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
        result_455.save(r'{}.tiff'.format(name455),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
        result_710.save(r'{}.tiff'.format(name710),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
        result_965.save(r'{}.tiff'.format(name965),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
                        
        
        print('[scan] Saved 4 timegated PSF', name)

    
        
        
    
    
    
    @pyqtSlot(bool)
    def toggle_lifetime(self, val): #TODO obsolete? use for FLIM
        if val == True:
            self.timegate = True
            
        if val == False:
            self.timegate = False
    
    
    @pyqtSlot(bool)
    def toggle_WideFieldshutter(self, val):
        """
        Connection: TO [frontend]  
        FROM [frontend, TCSPC]
        """ 
        if val is True:
            self.SetBitOn(self.BitTask7)
            
        elif val is False:
            self.SetBitOff(self.BitTask7)


#    @pyqtSlot(bool)
    def toggle_shutter_rg(self, val):
        if val is True:
            
            
            if self.shutter_state == False:
                self.shutter_state = True
                self.toggleShutterSignal.emit(True)   
                self.SetBitOn(self.BitTask1)
                self.SetBitOn(self.BitTask4)
    

        if val is False:
            
            
            if self.shutter_state == True:
                
                self.shutter_state = False
                self.toggleShutterSignal.emit(False)   

                self.SetBitOff(self.BitTask1)
                
                self.SetBitOff(self.BitTask4)
                


    @pyqtSlot(bool)
    def toggle_shutter(self, val):
        """
        Connection: TO [frontend]  
        FROM [frontend, TCSPC]
        """ 
        
        
        #Hacked
        if val is True:
            
            
            if self.shutter_state == False:
                
                self.shutter_state = True
                self.toggleShutterSignal.emit(True)        
#                self.SetBitOn(self.BitTask1)
                
                if self.midbeam == "200":
                    self.SetBitOn(self.BitTask1)
                elif self.midbeam == "455":
                    self.SetBitOn(self.BitTask2)
                elif self.midbeam == "710":
                    self.SetBitOn(self.BitTask3)
                elif self.midbeam == "965":
                    self.SetBitOn(self.BitTask4)
                
            
                print(datetime.now(), '[scan] Shutter opened')
            
        if val is False:
            
            
            if self.shutter_state == True:
                
                self.shutter_state = False
                self.toggleShutterSignal.emit(False)   
#                self.SetBitOff(self.BitTask1)
                
                
                if self.midbeam == "200":
                    self.SetBitOff(self.BitTask1)
                elif self.midbeam == "455":
                    self.SetBitOff(self.BitTask2)
                elif self.midbeam == "710":
                    self.SetBitOff(self.BitTask3)
                elif self.midbeam == "965":
                    self.SetBitOff(self.BitTask4)
                
                
                print(datetime.now(), '[scan] Shutter closed')
    

    
    @pyqtSlot(bool)
    def toggle_shutter_all(self, val):
        """
        Connection: TO [frontend]  
        FROM [frontend, TCSPC]
        """ 
        
        if val is True:
            
            
            if self.shutter_state == False:
                
                self.shutter_state = True
                self.toggleShutterSignal.emit(True)   
                
                
                
                self.SetBitOn(self.BitTask1)
                self.SetBitOn(self.BitTask2)
                self.SetBitOn(self.BitTask3)
                self.SetBitOn(self.BitTask4)
                self.SetBitOn(self.BitTask5)
                self.SetBitOn(self.BitTask6)
            
                print(datetime.now(), '[scan] Shutter opened')
            
        if val is False:
            
            
            if self.shutter_state == True:
                
                self.shutter_state = False
                self.toggleShutterSignal.emit(False)   
                self.SetBitOff(self.BitTask1)
                self.SetBitOff(self.BitTask2)
                self.SetBitOff(self.BitTask3)
                self.SetBitOff(self.BitTask4)
                self.SetBitOff(self.BitTask5)
                self.SetBitOff(self.BitTask6)
#                self.SetBitOff(self.BitTask5)
                
                print(datetime.now(), '[scan] Shutter closed')
    
    
    
    
    
    def SetBitOn(self, MyBitTask):
        '''Apply 5V output to a digital pin corresponding to the provided task'''
        MyBitTask.write(True)
#        MyBitTask.StopTask() # Task must be stopped in order to be able to play with the other channels/tasksß
        
    def SetBitOff(self, MyBitTask):
        '''Apply 0V output to a digital pin corresponding to the provided task'''    
        MyBitTask.write(False)
#        MyBitTask.StopTask()


#    @pyqtSlot(float)
    def setODscan(self):
        
        print("setting OD to ", self.scanOD)
        self.angle = self.scanOD / self.ODslope - self.ODangleoffset
        self.owisps10.moveAbsolute(1, int(self.angle))
        
        
    @pyqtSlot()
    def setODtcspc(self):
        
        print("setting OD to ", self.tcspcOD)
        self.angle = self.tcspcOD / self.ODslope - self.ODangleoffset
        self.owisps10.moveAbsolute(1, int(self.angle))  
        
        
    @pyqtSlot()
    def prebleach(self):
        #For PAINT internal dye
        #fixed OD -0.5
        print("setting OD to ", self.tcspcOD - 0.5)
        self.angle = (self.tcspcOD - 0.5) / self.ODslope - self.ODangleoffset
        self.owisps10.moveAbsolute(1, int(self.angle))  
        self.toggle_shutter_all(True)
        

    def emit_ROI_center(self):
        
        self.ROIcenterSignal.emit(self.ROIcenter)
        
        print('[scan] ROI center emitted')
    


    @pyqtSlot(np.ndarray)    
    def fit_center(self, clickposition):
        """ get rough center position from click (?)
            fit center and drive there 
        
        FROM Frontend FitandMoveSignal
        """
#        x = np.arange(self.initialPos[0], sizeg, pxg)
#        y = np.arange(0, sizeg, pxg)
        
        
        
        
        x0i = self.data_x[clickposition[0]]
        y0i = self.data_y[clickposition[1]]
#        
#          self.data_x = np.arange(self.initialPos[0], self.initialPos[0]+ self.scanRange,\
#                                self.scanRange / self.NofPixels)
#            self.data_y = np.arange(self.initialPos[1], self.initialPos[1] + self.scanRange,\
#                                self.scanRange / self.NofPixels) 
#            
        npx = 5#13
        
        
        index_start = clickposition - npx #make dynamic end of image!
        index_end = clickposition + npx +1 # +- 15 px?!
        
        print(clickposition, index_start)
        #TODO: x = x????
        x, y = np.meshgrid(self.data_x[int(index_start[0]):int(index_end[0])], self.data_y[int(index_start[1]):int(index_end[1])]) 
        psf_new = self.image[int(index_start[0]):int(index_end[0]), int(index_start[1]):int(index_end[1])] 
        
#        print(x.shape)
#        print(psf_new.shape)
        print(x0i,y0i)
#        print(psf_new)
        print(x[1])
        
        p0i = [x0i, y0i, 0, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt, cov = opt.curve_fit(poly_func, (x,y), psf_new.ravel(), p0 = p0i)    
        #TODO try and exception!
        
        
        
        print('found fit', aopt[0],aopt[1])
        
        #move to

        #after test
#        pos = (aopt[0:2]/ self.NofPixels* self.scanRange + self.initialPos[0:2]) 
        
#        sizeg = npx *2 +1
#        q = poly_func((x,y), *aopt)   
#        PD = np.reshape(q, (int(sizeg), int(sizeg)))
#        # find min value for each fitted function (EBP centers)
#        ind = np.unravel_index(np.argmin(PD[:, :], 
#            axis=None), PD[ :, :].shape)
#        print(ind)
        
#        print(index_start+ind)
#        self.found_fit_Signal.emit(index_start+ind)        
        
        
        #self.image[self.numberx] = np.flip(self.lineData)
#        print(aopt)
#        self.imageSignal.emit(image)
        pxpos = (aopt[0:2] - self.initialPos[0:2]) / self.scanRange * self.NofPixels
#        print()
        self.found_fit_Signal.emit(np.array(pxpos))
        pos = aopt[0:2]
        self.set_moveToXY(pos[0], pos[1])
        
        
        
        
        
        
    @pyqtSlot(np.ndarray)       
    def find_PickandDestroy(self, pxposition):
# =============================================================================
#         iterative?
#        1 additional scan -> fit center -> Destroy ? 
# =============================================================================
        #linedone == False
        
        
        #bla EVTL 6x6 Punkte, no kein Scan?!
        #iterative jede sekunde zentrieren

        
        self.pidevice.VEL("1", 10000)        
        self.pidevice.VEL("2", 10000)
        self.pidevice.VEL("3", 5000)
        
        self.PDNofPixel = 11 # per line
        self.PDmeasurementtime =  4  # ms # check conversions!
        self.PDscanrange = 100 * 10**-3 # um, 100nm # times 2.....
        
        #self.tcspcPrepareScanSignal.emit(self.PDmeasurementtime * self.PDNofPixel**2 * 1000) #time  needs to  be in us
        self.tcspcPrepareScanSignal.emit(self.PDmeasurementtime * 1000)
        
        
        
        print(pxposition)
        time.sleep(1) #done queue???
        position = [pxposition[0] * self.scanRange / self.NofPixels  + self.initialPos[0], \
                    (pxposition[1] ) * self.scanRange / self.NofPixels  + self.initialPos[1]]

#        print("trying to fit")
        aopt = self.scan_fit_PickandDestroy(position)
        
        print(aopt)
        midposx = aopt[0]
        midposy = aopt[1]
#        print("trying to fit2")
        MaximalFitDiscrepancy = 50 * 10**-3 # um - if larger rescan once
        SafetyShutdownFit = 2*self.PDscanrange # um # if larger - break
        
#        print("1. durchgang")
#        print(midposx, midposy, position)
#        if ((abs(midposx - position[0]) > MaximalFitDiscrepancy ) or (abs(midposy -  position[1]) > MaximalFitDiscrepancy)) \
#        and ((abs(midposx - position[0]) < SafetyShutdownFit )  or (abs(midposy -  position[1]) < SafetyShutdownFit)):
            #repeat once!
#            position[0], position[1] = [midposx, midposy]
#            midposx, midposy = self.scan_fit_PickandDestroy(position)
            
        
#        print("tried to fit")
        #TODO different pos?! earlier vs spaeter positions
        print(midposx, midposy)
        print(position)
        print(self.initialPos)
        self.tcspcCloseConnection.emit() 
        if (abs(midposx - position[0]) < MaximalFitDiscrepancy ) or (abs(midposy -  position[1]) < MaximalFitDiscrepancy):
            print("success")
            
            
#            pxposx = (midposx - self.initialPos[0]) / self.scanRange * self.NofPixels
#            pxposy = (midposy - self.initialPos[1]) / self.scanRange * self.NofPixels
#            midposy
#            midposx, midposy
            
            pos = (aopt[0:2]/ self.NofPixels* self.scanRange + self.initialPos[0:2]) 
            self.set_moveToXY(pos[0], pos[1])
            
            
            
            
            self.found_pick_PickandDestroy.emit(True)
            
        else:
            print("no success")
            self.found_pick_PickandDestroy.emit(False)


        
        
        

    def scan_fit_PickandDestroy(self, position):
        
        TCSPCdone = False
        
#        print("got find pick signal" )
        
        self.PickAndDestroylistx = np.arange(position[0] - self.PDscanrange, position[0] + self.PDscanrange + 10**-7, 2*self.PDscanrange/(self.PDNofPixel-1) )
        self.PickAndDestroylisty = np.arange(position[1] - self.PDscanrange, position[1] + self.PDscanrange + 10**-7, 2*self.PDscanrange/(self.PDNofPixel -1) )
        
#        print(self.PickAndDestroylistx.shape,self.PickAndDestroylisty.shape)
        print(self.PickAndDestroylistx)
        print(self.PickAndDestroylisty)
        self.set_moveToXY(self.PickAndDestroylistx[0],self.PickAndDestroylisty[0])
        self.toggle_shutter(True)
        
        
        
        counts = np.zeros((self.PDNofPixel,self.PDNofPixel))
        
        t0 = time.time()
        for i,xpos in enumerate(self.PickAndDestroylistx):
            for j, ypos in enumerate(self.PickAndDestroylisty):        
                self.set_moveToXY(xpos,ypos, t = 0)
                TCSPCdone = False
                self.tcspcMeasureScanPointSignal.emit()  #TODO: no TCSPC.... just len no export/convert
                
#                time.sleep( self.PDmeasurementtime*10**-3)
                
                while TCSPCdone == False:
                    try:
                        
                        TCSPCdone, countnr = self.scanTCSPCdata.get(timeout=0)
                        counts[i,j] = countnr
#                        counts.append(countsi)
                        
                        #print("x= ", self.pidevice.qPOS(1)[1],"y= ", self.pidevice.qPOS(2)[2])
                        #print("got data", TCSPCdone, count)
                    except queue.Empty:
        #                print("queue problems")
                        pass
                    
                
#                print(ypos)
#                pitools.waitontarget(self.pidevice, self.PDmeasurementtime)
#                pitools.waitontarget(self.pidevice, axes='1', polldelay = self.PDmeasurementtime) 
        t1 = time.time()
        self.toggle_shutter(False)
        
        
        print(t1-t0,t1,t0)
   
        print("scan bla quatsch 3")
        
        self.scanTCSPCdata.queue.clear()
        
#        print("got counts", count)
        #Bin abs photon times into into pixels
#        line_data = np.histogram(count, bins = self.PDNofPixel**2, range = (7, self.PDNofPixel**2 * self.PDmeasurementtime)) #count in ms
#        print("init",str( self.PickAndDestroylistx[int(self.PDNofPixel/2)]), str(self.PickAndDestroylisty[int(self.PDNofPixel/2)]))
#        (x,y) = np.meshgrid(self.PickAndDestroylistx, self.PickAndDestroylisty)
        
        
        image = counts.reshape(self.PDNofPixel,self.PDNofPixel)        
        length = image.shape[0]
        xs = np.arange(0,length)
        ys = np.arange(0,length)
        
        (x,y) = np.meshgrid(xs,ys)
        
        
        
#        p0i = [self.PickAndDestroylistx[int(self.PDNofPixel/2)], self.PickAndDestroylisty[int(self.PDNofPixel/2)], 0, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#               0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        p0i = [length/2,length/2, 0, 1 ,1, 1, 1, 1, 0, 0, 0]#        print(line_data)
#        print(line_data)
#        print(self.PDNofPixel)
#        print(line_data[0])
#        print(line_data[0].reshape(self.PDNofPixel,self.PDNofPixel))
        
#        print(line_data[0].reshape(self.PDNofPixel,self.PDNofPixel))
#        print(x)
#        print(self.PickAndDestroylistx)
#        print(self.PickAndDestroylistx.shape)
#        print(y)
#        print(len(x))
#        print(len(y))
        
#        image = line_data[0].reshape(self.PDNofPixel,self.PDNofPixel)
#        print("scan bla quatsch")
#        count = np.asarray(counts, dtype=np.float64)

        result = Image.fromarray(image.astype('uint16'))
        #added rotate 270 check
        result.rotate(270).save(r'{}.tiff'.format("test"),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
#        print("scan bla shape")
        print(x.shape, y.shape, image.shape)
        print(image)
        self.imageSignal.emit(image)
        #interpolation!
        

        try:
#            aopt, cov = opt.curve_fit(poly_func, (x,y), image.ravel(), p0 = p0i)
            aopt, cov = opt.curve_fit(poly_func2D, (x[::-1],y[::-1]), image.ravel(), p0 = p0i) 
            
        except(RuntimeError, ValueError):
            
            print(datetime.now(), '[SCAN] fit did not work')
            aopt = [0,0,0,0]
        print("tried fit", aopt)
        print("done fit")
        
        
        #        result.rotate(270).save(r'{}.tiff'.format("test"),  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) 
        
#        fit = poly_func2Dpl(*aopt)
        
        
#        
#        xs = np.arange(0,length+0.1,1)
#        ys = np.arange(0,length+0.1,1)
#        (x,y) = np.meshgrid(xs,ys)
#
#        
##        xs = np.arange(0,length,0.3)
##        ys = np.arange(0,length,0.3)
##        (x,y) = np.meshgrid(gridx,gridy)
#        
#        
        print("done fit2", aopt)      
#        Z = fit((x[::-1],y[::-1]))
#        length2 = x
        #plot fit
#        print("done fit3")
#        print(Z)
        
#        fig = plt.figure(figsize=(5, 5))
#        Z1 = np.reshape(Z, (-1, 2))
#        plt.pcolormesh(Z1)
#        plt.colorbar()
#        plt.show()
#        print("done fit4")      
#        ind1 = np.unravel_index(np.argmin(Z), (length2,length2))
##        print(ind1)
#        
#        xmin = x[-ind1[0],-ind1[1]]
#        ymin = y[-ind1[0],-ind1[1]]
        
        posmin = aopt[0:2]
        print("new", posmin)
#        pos = (aopt[0:2]/ self.PDNofPixel * self.PDscanrange + self.initialPos[0:2]) 
            
#
#        sizeg = self.PDNofPixel
#        q = poly_func((x,y), *aopt)   
#        PD = np.reshape(q, (int(sizeg), int(sizeg)))
#        # find min value for each fitted function (EBP centers)
#        ind = np.unravel_index(np.argmin(PD[:, :], 
#            axis=None), PD[ :, :].shape)
#        
#        #self.image[self.numberx] = np.flip(self.lineData)
#        print(aopt)
        

        #2*self.PDscanrange/(self.PDNofPixel-1) 
        
#        pxpos = np.array([])
        
#        pxpos = (np.array(posmin) - np.array(position) + self.PDscanrange ) /(2*self.PDscanrange/(self.PDNofPixel-1))##
        ##plot fit
        
#        fit = poly_func2D(*fit)
    
#        Z = fit(*np.indices(np.array((x[::-1],y[::-1])).shape))
        
        #plot fit
#        
#        fig = plt.figure(figsize=(5, 5))
#        plt.pcolormesh(fit)
#        plt.colorbar()
#        plt.show()
#        
        
#        print(pos)
#        print(posmin)
        realpos = posmin *(2*self.PDscanrange/(self.PDNofPixel-1)) +  np.array(position) - self.PDscanrange
        print(realpos)
#        print(ind)
        self.found_fit_Signal.emit(posmin)
        
        

#        midposx = aopt[0]
#        midposy = aopt[1]
        
        return np.array(realpos) #midposx, midposy
    
    @pyqtSlot()
    def sendImagetoPD(self):
        
        self.imageSignalPD.emit(self.image)       
                      
    def make_connection(self, frontend):
        
        
        
        frontend.liveviewSignal.connect(self.liveview)
        frontend.moveToROIcenterButton.clicked.connect(self.moveTo_roi_center)
        frontend.currentFrameButton.clicked.connect(self.save_current_frame)
        frontend.saveReltimesButton.clicked.connect(self.save_current_TCSPC)
        
        frontend.moveToButton.clicked.connect(self.moveTo_action)
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.closeSignal.connect(self.stop)
        frontend.FitandMoveSignal.connect(self.fit_center)
        #TODO: Pickand Destroy
        #frontend.PickandDestroySignal.connect(self.init_PickandDestroy)
        
        frontend.shutterButton.clicked.connect(lambda: self.toggle_shutter(frontend.shutterButton.isChecked()))
#        frontend.flipperButton.clicked.connect(lambda: self.toggle_flipper(frontend.flipperButton.isChecked()))
#        frontend.lifetimeButton.clicked.connect(lambda: self.toggle_lifetime(frontend.lifetimeButton.isChecked())) #TODO: check if 
        
        frontend.xUpButton.pressed.connect(lambda: self.relative_move('x', 'up'))
        frontend.xDownButton.pressed.connect(lambda: self.relative_move('x', 'down'))
        frontend.yUpButton.pressed.connect(lambda: self.relative_move('y', 'up'))
        frontend.yDownButton.pressed.connect(lambda: self.relative_move('y', 'down'))        
        frontend.zUpButton.pressed.connect(lambda: self.relative_move('z', 'up'))
        frontend.zDownButton.pressed.connect(lambda: self.relative_move('z', 'down'))
          
    def stop(self):


        self.toggle_shutter(False)
        self.liveview_stop()
        
        

        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0
        self.pidevice.VEL("1", 10000)
        self.pidevice.VEL("2", 10000)
        self.pidevice.VEL("3", 5000)
        #self.moveTo(x_0, y_0, z_0)
        self.pidevice.CloseConnection()     
        
        
if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    scanTCSPCdata = queue.Queue()
    
    owisps10 = owis.PS10("\\drivers\\PS10-1_2.owi")
    
    CONTROLLERNAME = 'E-727'
    STAGES = ('P-733.3CD',)
    REFMODES = ['FNL', ]
    #init stage
    
    with GCSDevice(CONTROLLERNAME) as pidevice:
        #pidevice.InterfaceSetupDlg()
        pidevice.ConnectUSB(serialnum='0118038033')
        pidevice.send("SVO 1 1")
        pidevice.send("SVO 2 1")
        pidevice.send("SVO 3 1")
        if pidevice.HasqVER():
            print('version info:\n{}'.format(pidevice.qVER().strip()))
        time.sleep(1)
        #pitools.startup(pidevice, stages=STAGES,  refmode=REFMODES)
        #setupDevice(pidevice)
    
        worker = Backend(pidevice, owisps10 , scanTCSPCdata)    
        gui = Frontend()
    
        worker.make_connection(gui)
        gui.make_connection(worker)
    
        gui.emit_param()
        worker.emit_param()
    
#
        scanThread = QtCore.QThread()
        worker.moveToThread(scanThread)
        worker.viewtimer.moveToThread(scanThread)
        worker.viewtimer.timeout.connect(worker.update_view)
    
        scanThread.start()

    
        gui.setWindowTitle('scan')
        gui.show()
        
        app.exec_()
