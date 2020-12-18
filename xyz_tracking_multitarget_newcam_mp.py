# -*- coding: utf-8 -*-
"""
Created on 05/06/19 

@author: Jonas Zaehringer
original xy tracking script and focus script: Luciano Masullo

"""

import numpy as np
import time
import scipy.ndimage as ndi
import scipy.stats as stats
import ctypes as ct
from datetime import date, datetime
import os
import sys


import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
from scipy import optimize as opt
from PIL import Image

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools
#import tools.pi as pi
#import scan

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
import qdarkstyle

from lantz.drivers.andor import ccd 
from pipython import GCSDevice, pitools
#from instrumental.drivers.cameras import uc480
#from instrumental import instrument, list_instruments

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from thorlabs_windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

    
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
import threading

#from pathos.multiprocessing import _Pool as Pool
import multiprocessing as mp
from multiprocessing import Pool


try:
    #  For Python 2.7 queue is named Queue
    import Queue as queue
except ImportError:
    import queue
DEBUG = 'True'


def zMoveTo(pidevice, z_f):
    pidevice.MOV('3', z_f)
    time.sleep(0.05)
    #start = time.time()
    #pitools.waitontarget(pidevice, axes='3')
    #end = time.time()
    #
    position = pidevice.qPOS(3)[3]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
    if DEBUG:
        #print(end-start)
        print("target position is {}".format(z_f))
        print('current position of axis {} is {:.2f}'.format(3, position))
    #actuatorParameters(pidevice, z_f)


def XMoveTo(pidevice, X_f):
    pidevice.MOV('1', X_f)
    time.sleep(0.05)
    #start = time.time()
    #pitools.waitontarget(pidevice, axes='3')
    #end = time.time()
    #
    position = pidevice.qPOS(1)[1]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
    if DEBUG:
        #print(end-start)
        print("target position is {}".format(X_f))
        print('current position of axis {} is {:.2f}'.format(1, position))
    #actuatorParameters(pidevice, z_f)



def YMoveTo(pidevice, Y_f):
    pidevice.MOV('2', Y_f)
    time.sleep(0.05)
    #start = time.time()
    #pitools.waitontarget(pidevice, axes='3')
    #end = time.time()
    #
    position = pidevice.qPOS(2)[2]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
    if DEBUG:
        #print(end-start)
        print("target position is {}".format(Y_f))
        print('current position of axis {} is {:.2f}'.format(1, position))
    #actuatorParameters(pidevice, z_f)


def ztrack_mp_routine(image, zROIcoordinates):
#    print("here")    
#    print("ztrack I'm running on CPU #%s" % mp.current_process().name)
#    image, zROIcoordinates = helper[0], helper[1]
#    print(zROIcoordinates)
    
#    print("t1")
#    print(helper)
#    print(helper.shape)
#    print(image)
    
    subimagez = image[zROIcoordinates[0]:zROIcoordinates[1],zROIcoordinates[2]:zROIcoordinates[3]]
    #part
#    print("t2")
    massCenter =  np.array(ndi.measurements.center_of_mass(subimagez))
    
#    print(massCenster)
    return massCenter  


def xytrack_mp_routine(image, ROIcoordinates, pxSize):
#    print("xytrack I'm running on CPU #%s" % mp.current_process().name)
    currentx = []
    currenty = []
#    roinumber = 0   
    for roicoordinate in ROIcoordinates:
        
        xmin, xmax, ymin, ymax = roicoordinate
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = roicoordinate * pxSize
        
        # select the data of the image corresponding to the ROI

        array = image[xmin:xmax, ymin:ymax]
        
        # set new reference frame
        
        xrange_nm = xmax_nm - xmin_nm
        yrange_nm = ymax_nm - ymin_nm
             
        x_nm = np.arange(0, xrange_nm, pxSize)
        y_nm = np.arange(0, yrange_nm, pxSize)
        
        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
        
        # find max 
        
        argmax = np.unravel_index(np.argmax(array, axis=None), array.shape)
        #print("argmax")
        #print(argmax)
        x_center_id = argmax[0]
        y_center_id = argmax[1]
        
        # define area around maximum
    
        xrange = 13 # in px
        yrange = 13 # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')
        
        x_sub_nm = np.arange(0, xsubsize) * pxSize
        y_sub_nm = np.arange(0, ysubsize) * pxSize

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
        # make initial guess for parameters
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 50 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
        try:
            poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                     array_sub.ravel(), p0=initial_guess_G)
            
        except(RuntimeError, ValueError):
            
            print(datetime.now(), '[xyz_tracking] Gaussian fit did not work')
#            toggle_feedback(False) #todo off!!#######

        
        # retrieve results

        poptG = np.around(poptG, 2)
    
        A, x0, y0, σ_x, σ_y, bkg = poptG
        currentx.append(x0 + Mx_nm[xmin_id, ymin_id])
        currenty.append(y0 + My_nm[xmin_id, ymin_id])
#            print(Mx_nm[xmin_id, ymin_id])
#            print(currentx[roinumber])
#        roinumber = roinumber + 1
#    print(np.array(currentx), np.array(currenty))
    return np.array(currentx), np.array(currenty)

    

class Frontend(QtGui.QFrame):
    
    liveviewSignal = pyqtSignal(bool)
    roiInfoSignal = pyqtSignal(int, np.ndarray) 
    zroiInfoSignal = pyqtSignal(int, np.ndarray)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    
    """
    Signals
    
    - liveviewSignal:
         To: [backend] liveview
             
    - roiInfoSignal:
         To: [backend] get_roi_info
        
    - closeSignal:
         To: [backend] stop
        
    - saveDataSignal:
         To: [backend] get_save_data_state
        
    """
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
        # initial ROI parameters        
        
        self.NofPixels = 200
        self.roi = None
        self.ROInumber = 0
        self.roilist = []
        
        self.zroi = None
        self.zROInumber = 0
        self.zroilist = []

    def create_roi(self):
        
        ROIpen = pg.mkPen(color='r')

        ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
        self.roi = viewbox_tools.ROI2(self.NofPixels/2, self.vb, ROIpos,
                                     handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
        
        self.ROInumber += 1
        
        self.roilist.append(self.roi)
        
        self.ROIButton.setChecked(False)

#        else:
#
#            self.vb.removeItem(self.roi)
#            self.roi.hide()
    
    
    def create_zroi(self):
        
        ROIpen = pg.mkPen(color='y')

        ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
        self.zroi = viewbox_tools.ROI2(self.NofPixels/2, self.vb, ROIpos,
                                     handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
        
        self.zROInumber += 1
        
        self.zroilist.append(self.zroi)
        
        self.zROIButton.setChecked(False)

#        else:
#
#            self.vb.removeItem(self.roi)
#            self.roi.hide()
    
    
    def emit_roi_info(self):
        
#        print('Set ROIs function')
        
        roinumber = len(self.roilist)
        roicoordinates = np.zeros((len(self.roilist),4))
        if roinumber == 0:
            
            print(datetime.now(), '[xyz_tracking] Please select a valid ROI for beads tracking')
            
        else:
            
            coordinates = np.zeros((4))
            
            for i in range(len(self.roilist)):
                
    #            print(self.roilist[i].pos())
    #            print(self.roilist[i].size())
                xmin, ymin = self.roilist[i].pos()
                xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
        
                coordinates = np.array([xmin, xmax, ymin, ymax])  
                roicoordinates[i] = coordinates
                
            self.roiInfoSignal.emit(roinumber, roicoordinates)
            print("emit ROI Info: ")
            print(roinumber)
            print(roicoordinates)
#            self.roiInfoSignal.emit(roinumber, coordinates)

    def emit_zroi_info(self):
        
#        print('Set ROIs function')
        
        zroinumber = len(self.zroilist)
        
        if zroinumber == 0:
            
            print(datetime.now(), '[focus] Please select a valid ROI for beads tracking')
            
        else:
            
            coordinates = np.zeros((4))
            
            for i in range(len(self.zroilist)):
                
    #            print(self.roilist[i].pos())
    #            print(self.roilist[i].size())
                xmin, ymin = self.zroilist[i].pos()
                xmax, ymax = self.zroilist[i].pos() + self.zroilist[i].size()
        
                coordinates = np.array([xmin, xmax, ymin, ymax])  
#            roicoordinates[i] = coordinates
                
#            self.roiInfoSignal.emit(roinumber, roicoordinates)
            self.zroiInfoSignal.emit(zroinumber, coordinates)



    def delete_roi(self):
        
        for i in range(len(self.roilist)):
            
            self.vb.removeItem(self.roilist[i])
            self.roilist[i].hide()
            
        self.roilist = []
        self.delete_roiButton.setChecked(False)
        self.ROInumber = 0
        
    def delete_zroi(self):
        
        for i in range(len(self.zroilist)):
            
            self.vb.removeItem(self.zroilist[i])
            self.zroilist[i].hide()
            
        self.zroilist = []
        self.delete_zroiButton.setChecked(False)
        self.zROInumber = 0
        
        
        
    def toggle_liveview(self):
        
        if self.liveviewButton.isChecked():
            
            self.liveviewSignal.emit(True)
        
        else:
            
            self.liveviewSignal.emit(False)
            self.liveviewButton.setChecked(False)
            self.emit_roi_info()
            self.img.setImage(np.zeros((1024,1024)), autoLevels=False)
            print(datetime.now(), '[xyz_tracking] Live view stopped')
        
    @pyqtSlot()  
    def get_roi_request(self):
        
        print(datetime.now(), '[xyz_tracking] got ROI request')
        
        self.emit_roi_info()
        
        
    @pyqtSlot()  
    def get_zroi_request(self):
        
        print(datetime.now(), '[focus] got ROI request')
        
        self.emit_zroi_info()
        
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        
#        if DEBUG:
#            print(datetime.now(),'[xyz_tracking-frontend] got image signal')

#        self.img.setImage(img, autoLevels=False) 
        self.img.setImage(img, levels = [0,1024]) 
        

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, time, xData, yData, ztime, zData):

        self.xCurve.setData(time, xData)
        self.yCurve.setData(time, yData)
        self.zCurve.setData(ztime, zData)
        
        self.xyDataItem.setData(xData, yData)
        
        if len(xData) > 2:
            
            self.plot_ellipse(xData, yData)
        
    def plot_ellipse(self, x_array, y_array):
        
        pass
        
#            cov = np.cov(x_array, y_array)
#            
#            a, b, theta = tools.cov_ellipse(cov, q=.683)
#            
#            theta = theta + np.pi/2            
##            print(a, b, theta)
#            
#            xmean = np.mean(xData)
#            ymean = np.mean(yData)
#            
#            t = np.linspace(0, 2 * np.pi, 1000)
#            
#            c, s = np.cos(theta), np.sin(theta)
#            R = np.array(((c, -s), (s, c)))
#            
#            coord = np.array([a * np.cos(t), b * np.sin(t)])
#            
#            coord_rot = np.dot(R, coord)
#            
#            x = coord_rot[0] + xmean
#            y = coord_rot[1] + ymean
            
            # TO DO: fix plot of ellipse
            
#            self.xyDataEllipse.setData(x, y)
#            self.xyDataMean.setData([xmean], [ymean])
        
    @pyqtSlot(bool, bool, bool, bool, bool, bool)
    def get_backend_states(self, tracking, ztracking, feedback, zfeedback, savedata, invertImage):
        
#        print(datetime.now(), '[xyz_tracking] Got backend states')
        
        if tracking is True:
            
            self.trackingBeadsBox.setChecked(True)
        
        if tracking is False:
            
            self.trackingBeadsBox.setChecked(False)
            
        if feedback is True:
            
            self.feedbackLoopBox.setChecked(True)
            
        if feedback is False:
            
            self.feedbackLoopBox.setChecked(False)
        
        if ztracking is True:
            
            self.ztrackingBeadsBox.setChecked(True)
        
        if ztracking is False:
            
            self.ztrackingBeadsBox.setChecked(False)
            
        if zfeedback is True:
            
            self.zfeedbackLoopBox.setChecked(True)
            
        if zfeedback is False:
            
            self.zfeedbackLoopBox.setChecked(False) 
        
        
        if savedata is True:
            
            self.saveDataBox.setChecked(True)
            
        if savedata is False:
            
            self.saveDataBox.setChecked(False)
            
            
        if invertImage is True:
            
            self.invertImageBox.setChecked(True)
        
        if invertImage is False:
            
            self.invertImageBox.setChecked(False)

    def emit_save_data_state(self):
        
        if self.saveDataBox.isChecked():
            
            self.saveDataSignal.emit(True)
            self.emit_roi_info()
            
        else:
            
            self.saveDataSignal.emit(False)
    

    
    
    def make_connection(self, backend):
            
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        backend.updateGUIcheckboxSignal.connect(self.get_backend_states)
        backend.getROISignal.connect(self.emit_roi_info)
        backend.getROISignal.connect(self.emit_zroi_info)


    def setup_gui(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(200)
        self.paramWidget.setFixedWidth(250)
        
        grid.addWidget(self.paramWidget, 0, 1)
        
        # image widget layout
        
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(350) # von 350 auf 550
        imageWidget.setMinimumWidth(350)
        
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)
        grid.addWidget(imageWidget, 0, 0)
        
        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        lut = viewbox_tools.generatePgColormap(cmaps.parula)
        self.hist.gradient.setColorMap(lut)
#        self.hist.vb.setLimits(yMin=800, yMax=3000)

        ## TO DO: fix histogram range


        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # xy drift graph (graph without a fixed range)
        
        self.xyGraph = pg.GraphicsWindow()
#        self.xyGraph.resize(200, 300)
        self.xyGraph.setAntialiasing(True)
        
        self.xyGraph.statistics = pg.LabelItem(justify='right')
        self.xyGraph.addItem(self.xyGraph.statistics)
        self.xyGraph.statistics.setText('---')
        
        self.xyGraph.xPlot = self.xyGraph.addPlot(row=1, col=0)
        self.xyGraph.xPlot.setLabels(bottom=('Time', 's'),
                            left=('Y position', 'nm'))   # TO DO: clean-up the x-y mess (they're interchanged)
        self.xyGraph.xPlot.showGrid(x=True, y=True)
        self.xCurve = self.xyGraph.xPlot.plot(pen='b')
        
        self.xyGraph.yPlot = self.xyGraph.addPlot(row=0, col=0)
        self.xyGraph.yPlot.setLabels(bottom=('Time', 's'),
                                     left=('X position', 'nm'))
        self.xyGraph.yPlot.showGrid(x=True, y=True)
        self.yCurve = self.xyGraph.yPlot.plot(pen='r')
        
        
        #addded
        self.xyGraph.zPlot = self.xyGraph.addPlot(row=2, col=0)
        self.xyGraph.zPlot.setLabels(bottom=('Time', 's'),
                            left=('Z position', 'nm'))   # TO DO: clean-up the x-y mess (they're interchanged)
        self.xyGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyGraph.zPlot.plot(pen='y')
        
        # xy drift graph (2D point plot)
        
        self.xyPoint = pg.GraphicsWindow()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)
        
#        self.xyPoint.xyPointPlot = self.xyGraph.addPlot(col=1)
#        self.xyPoint.xyPointPlot.showGrid(x=True, y=True)
        
        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(bottom=('X position', 'nm'),
                                  left=('Y position', 'nm'))
        
        self.xyDataItem = self.xyplotItem.plot([], pen=None, symbolBrush=(255,0,0), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataMean = self.xyplotItem.plot([], pen=None, symbolBrush=(117, 184, 200), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataEllipse = self.xyplotItem.plot(pen=(117, 184, 200))

        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.clicked.connect(self.toggle_liveview)
        
        # create ROI button
    
        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.create_roi)
        
        self.calibrationButton = QtGui.QPushButton('Z Calibrate')
        
        
        
        self.xcalibrationButton = QtGui.QPushButton('X Calibrate')
        
        # create ROI button
    
        self.zROIButton = QtGui.QPushButton('Z ROI')
        self.zROIButton.setCheckable(True)
        self.zROIButton.clicked.connect(self.create_zroi)
        
        # select ROI
        
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        self.selectROIbutton.clicked.connect(self.emit_roi_info)
        
        
        # select z ROI
        
        self.selectzROIbutton = QtGui.QPushButton('Select Z ROI')
        self.selectzROIbutton.clicked.connect(self.emit_zroi_info)
        
        
        # delete ROI button
        
        self.delete_roiButton = QtGui.QPushButton('delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
        
        
         # delete ROI button
        
        self.delete_zroiButton = QtGui.QPushButton('delete Z ROIs')
        self.delete_zroiButton.clicked.connect(self.delete_zroi)
        
        # position tracking checkbox
        
        self.exportDataButton = QtGui.QPushButton('export current data')
        
        self.exportzDataButton = QtGui.QPushButton('export current z data')

        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('Track beads')
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        
    
        # position tracking checkbox
        
        self.ztrackingBeadsBox = QtGui.QCheckBox('Track Focus')
        self.ztrackingBeadsBox.stateChanged.connect(self.emit_zroi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')


        # turn ON/OFF zfeedback loop
        
        self.zfeedbackLoopBox = QtGui.QCheckBox('Focus Feedback loop')

        # Invert Image for xy tracking
        self.invertImageBox = QtGui.QCheckBox('Invert Image')

        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        
        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.ROIButton, 1, 0)
        subgrid.addWidget(self.zROIButton, 1, 3)
        subgrid.addWidget(self.selectROIbutton, 2, 0)
        subgrid.addWidget(self.selectzROIbutton, 2, 3)
        subgrid.addWidget(self.delete_roiButton, 3, 0)
        subgrid.addWidget(self.delete_zroiButton, 3, 3)
        subgrid.addWidget(self.exportDataButton, 4, 0)
        subgrid.addWidget(self.clearDataButton, 5, 3)
        subgrid.addWidget(self.trackingBeadsBox, 0, 1)
        subgrid.addWidget(self.feedbackLoopBox, 1, 1)
        subgrid.addWidget(self.ztrackingBeadsBox, 2, 1)
        subgrid.addWidget(self.zfeedbackLoopBox, 3, 1)
        subgrid.addWidget(self.invertImageBox, 4, 1)
        
        subgrid.addWidget(self.saveDataBox, 5, 1)
        subgrid.addWidget(self.calibrationButton, 0, 3)
        subgrid.addWidget(self.xcalibrationButton, 5, 0)
        subgrid.addWidget(self.exportzDataButton, 4, 3)
#        subgrid.addWidget(self.tcspcFeedbackBox, 7, 0)  # TO DO: put this in the tcpsc widget (plus signals, etc) and not here

        
        grid.addWidget(self.xyGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1)
        
        
    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        
        super().closeEvent(*args, **kwargs)
        
    
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    updateGUIcheckboxSignal = pyqtSignal(bool, bool, bool, bool, bool, bool)
    #TODO: zisdone??
    getROISignal= pyqtSignal()
    
    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    zIsDone = pyqtSignal(bool, float)
    """
    Signals
    
    - changedImage:
        To: [frontend] get_image
             
    - changedData:
        To: [frontend] get_data
        
    - updateGUIcheckboxSignal:
        To: [frontend] get_backend_states
        
    - xyIsDone:
        To: [psf] get_xy_is_done

    """

    def __init__(self, camera, pidevice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """
        self.andor = andor
        self.adw = adw
        
        self.setup_camera()"""
        
        self.camera = camera
        self.pidevice = pidevice
        self.initialize_camera()
        # folder
        
        today = str(date.today()).replace('-', '')  # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'\xyzdata'
        self.filename = folder + filename
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.update)

        
        
        self.tracking_value = False
        self.ztracking_value = False

        self.save_data_state = False
        self.feedback_active = False
        self.zfeedback_active = False
        
        self.numberofrois = 1
        self.refresh = 2
        self.feedbackrefresh = 10
        
        self.ROIcoordinates = []
        self.idletime = 300
        self.measurementtime = 50
        
        self.invertImage = False

        self.npoints = 300 #maybe increase for measurement
        self.buffersize = 50000
        
        self.currentx = np.zeros(self.numberofrois)
        self.currenty = np.zeros(self.numberofrois)
        
        self.reset()
        self.reset_data_arrays()
        
        
        self.counter = 0
        self.pool = Pool(processes=6)
   
        
    def initialize_camera(self):
        
        self._image_width = self.camera.image_width_pixels
        self._image_height = self.camera.image_height_pixels
        self._bit_depth = self.camera.bit_depth
        self.camera.image_poll_timeout_ms = 1
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
#        rawimage = self.camera.latest_frame()
        #image = np.sum(rawimage, axis=2)
#        image = rawimage
        
        self.pxSize = 44 / 0.955 * 0.66/ 1.1 / 0.85 / 1.11#71 x60 ob  # 83   # in nm, TO DO: change for an input from the user
        self.zpxSize = 17.7 / 0.165 /3.76 / 0.855 /0.52 * 0.66 / 0.7 /1.23
        self.sensorSize = (self._image_width,self._image_height)
        self.focusSignal = 0
#        print(self.sensorSize)
        # set focus update rate
        
        #self.scansPerS = 20

        #self.viewTime = 1000 / self.scansPerS
        #self.viewTimer = QtCore.QTimer()
        

    
    @pyqtSlot(bool)
    def liveview(self, value):
        
        '''
        Connection: [frontend] liveviewSignal
        Description: toggles start/stop the liveview of the camera.
        
        '''
        
        

        if value:
            self.liveview_start()

        else:
            self.liveview_stop()

        
    def liveview_start(self):
        
        self.initial = True
        
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] Liveview start')
        #print(datetime.now(), '[xyz_tracking] Andor temperature status:', self.andor.temperature_status)

        # Initial image
#        
#        try:
#            self.camera.stop_live_video()
#
#            
#        except: # TODO: change for specific Error
#            
#            pass
        self.camera.exposure_time_us = 15000
        time.sleep(0.3)
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        self.framenumber = 0
        
        self.camera.gain = 30
        
        
#        self.camera.start_live_video(framerate='15 Hz')
#        self.camera.master_gain = 1
        
#        self.camera.exposure_time = "50ms"
          
        frame = self.camera.get_pending_frame_or_null() #TODO: subsampling - 
#        print(self.image)
        if frame is not None:
            self.image = frame
            self.changedImage.emit(self.image)

        self.viewtimer.start(self.idletime) # DON'T USE time.sleep() inside the update()
                                  # 400 ms ~ acq time + gaussian fit time
    
    def liveview_stop(self):
        
        self.viewTimer.stop()
        
        x0 = 0
        y0 = 0
        x1 = 1280 
        y1 = 1024 
            
        val = np.array([x0, y0, x1, y1])
        self.camera._set_AOI(*val)
    
    
    """programme ausmisten
    def set_actuator_param(self ):
        
        z_f = 5
        self.pidevice.MOV('3', z_f)"""
        
        
    def actuator_z(self, z_f):
        print("moving to z = {}".format(z_f))
        self.pidevice.MOV('3', z_f)
        #time.sleep(0.05)
        #self.viewtimer.start(400) #TODO: sleep?
            #start = time.time()
            #pitools.waitontarget(pidevice, axes='3')
        #end = time.time()
        #
        position = self.pidevice.qPOS(3)[3]  # query single axis
        # position = pidevice.qPOS()[str(axis)] # query all axes
        if DEBUG:
            #print(end-start)
            print("target position is {}".format(z_f))
            print('current position of axis {} is {:.2f}'.format(3, position))
            #actuatorParameters(pidevice, z_f)

    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'):
        ''' 
        Connection: [frontend] feedbackLoopBox.stateChanged
        Description: toggles ON/OFF feedback for either continous (TCSPC) 
        or discrete (scan imaging) correction
        '''
        
        if val is True:
            
            self.feedback_active = True

            # set up and start actuator process
            
            if mode == 'continous':
                #TODO:
                pass
                #self.set_actuator_param()
                #self.adw.Start_Process(4)
            
            if DEBUG:
                print(datetime.now(), '[xyz_tracking] Feedback loop ON')
            
        if val is False:
            
            self.feedback_active = False
            
            if DEBUG:
                print(datetime.now(), '[xyz_tracking] Feedback loop OFF')
            
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, self.zfeedback_active, 
                                          self.save_data_state, self.invertImage) 
    
    #z Feedback
    @pyqtSlot(bool)
    def toggle_zfeedback(self, val, mode='continous'):
        ''' Toggles ON/OFF z-feedback for either continous (TCSPC) 
        or discrete (scan imaging) correction'''
        print("zfeed")

        if val is True:
            
#            self.reset() 
            self.setup_zfeedback()
            self.update() 
            self.zfeedback_active = True
            print("turning feedback on really")
            # set up and start actuator process
            
            #if mode == 'continous':
            
                #self.set_actuator_param() #needed?
                #self.adw.Start_Process(3)
            
            print(datetime.now(), ' [focus] Feedback loop ON')
            
        if val is False:
            
            self.zfeedback_active = False
            print(datetime.now(), ' [focus] Feedback loop OFF')
    
    
    
    
    
    @pyqtSlot()    
    def setup_zfeedback(self):
        
        ''' set up on/off zfeedback loop'''
        
        print(datetime.now(), '[focus] feedback setup 0')

        #self.setPoint = self.z * self.pxSize # define setpoint
        self.setPoint = self.z # define setpoint
        initial_z = self.pidevice.qPOS('3')['3'] # current z position of the piezo
        self.target_z = initial_z # set initial_z as target_z
        
        if DEBUG:
            print("feedback initialized at {}".format(initial_z))
        
        #self.changedSetPoint.emit(self.focusSignal) #TODO: change?
        
        print(datetime.now(), '[focus] feedback setup 1')

        # TO DO: implement calibrated version of this
    
    
    def update_zfeedback(self, mode='continous'):
        #TODO: needed? Signal connect?
        
        #dz = self.focusSignal * self.pxSize - self.setPoint
        dz = self.z  - self.setPoint

#        print('dz', dz, ' nm')
        
        threshold = 7 # in nm
        far_threshold = 20 # in nm
        correct_factor = 1
        security_thr = 200 # in nm
        
        if np.abs(dz) > threshold:
            
            if np.abs(dz) < far_threshold:
                
                dz = correct_factor * dz
    
        if np.abs(dz) > security_thr:
            
            print(datetime.now(), '[focus] Correction movement larger than 200 nm, active correction turned OFF')
            
        else:
            
            self.target_z = self.target_z + dz/1000  # conversion to µm
            
            if mode is 'continous':
                
                self.actuator_z(self.target_z)
                
            if mode is 'discrete':
                print(datetime.now(), '[focus] discrete correction to', self.target_z)
                # it's enough to have saved the value self.target_z
                
    #TODO: update_stats FOCUS
    def update(self):
        """ General update method """
        #print(datetime.now(), '[xyz_tracking] entered update')
        
        self.update_view()

        if self.tracking_value:
                
#            self.track()
#            self.update_graph_data()
            self.pool.apply_async(xytrack_mp_routine,  args = [self.image,self.ROIcoordinates, self.pxSize], callback = self.xycallback)
            
            
            if self.feedback_active and self.framenumber % self.feedbackrefresh == 0:
                    
                self.correct()
                
        if self.ztracking_value:
#            print("tracking")
#            print(self.pool)
#            print(self.ztrackhelperroutine(self.image, self.zROIcoordinates))
#            print("modulle I'm running on CPU #%s" % mp.current_process().name)
            
            self.pool.apply_async(ztrack_mp_routine,  args = [self.image,self.zROIcoordinates], callback = self.zcallback)
            
#            
#            print("finished?!")
            #            self.ztrack()
#            self.update_graph_data()
            if self.zfeedback_active and self.framenumber % self.feedbackrefresh == 0:
                    
                self.zcorrect()
                         
        self.counter += 1  # counter to check how many times this function is executed
        
        
    def zcallback(self,result):
#        print("callback I'm running on CPU #%s" % mp.current_process().name)
#        print("callllback#################")
              
        self.massCenter = result
#        print(self.massCenter)
        #new signal?
        self.ztrack()
#                if self.ztracking_value:
        self.update_graph_data()

 
    def xycallback(self, results):
        
        
#        print("xycallback I'm running on CPU #%s" % mp.current_process().name)
        
#        print(results)
        
        self.currentx = results[0]
        self.currenty = results[1]
        self.track()
        
        if self.ztracking_value != True:
            self.update_graph_data()
        
 
            
    def update_view(self):
        """ Image update while in Liveview mode """
        means = False
        
        """
        sampling = True
        sample = [0,-1,-1,-1]
        numframe = 0"""
        maxframes = 4
        tempimage = np.zeros(self.sensorSize)
        """
        if means:
            #while sampling:
            while numframe < 4:
                #teimage = self.camera.latest_frame()
                #time.sleep(0.1)
                teimage = self.camera.grab_image(n_frames = 4)
                
                #print(teimage)
                #print(teimage[0][0::4])
                #if teimage[0][0::4] != sample and numframe < maxframes:
                numframe += 1
                    #sample = teimage[0,0::4] # check if same frame not needed? -> test
                tempimage += teimage
                """"""elif teimage[0,0::4] != sample and numframe == maxframes:
                    numframe += 1
                    tempimage += teimage
                    sampling = False
                """ """
            #self.image = [x / (numframe) for x in tempimage]
            self.image = tempimage / maxframes
        else: 
            self.image = self.camera.latest_frame()"""
        #print(self.image)
        if means:
            testimage = self.camera.grab_image(n_frames = maxframes)
            for i in testimage:
                tempimage += i
            self.image = tempimage
        else:
#            
            #self.image = 255 - self.camera.latest_frame()
#            print(self.image)
            frame = self.camera.get_pending_frame_or_null() 
                        
            if frame is not None:
                self.image = frame.image_buffer #>> (self._bit_depth - 8)
#                print(self.image)
                
                
                if self.framenumber % self.refresh:
                    
                    self.changedImage.emit(self.image)
#                self.changedImage.emit(self.image)
                self.framenumber = self.framenumber + 1
            else:
                pass
#                print("None")
            
    def update_graph_data(self):

        if self.ptr < self.npoints:
            self.xData[self.ptr] = self.x
            self.yData[self.ptr] = self.y
            self.zData[self.ptr] = self.z #focus
            self.time[self.ptr] = self.currentTime
            self.ztime[self.ptr] = self.zcurrentTime
            
            
            if self.framenumber % 4:
                self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.xData[0:self.ptr + 1],
                                  self.yData[0:self.ptr + 1], self.ztime[0:self.ptr + 1], self.zData[0:self.ptr + 1])

        else:
            self.xData[:-1] = self.xData[1:]
            self.xData[-1] = self.x
            self.yData[:-1] = self.yData[1:]
            self.yData[-1] = self.y
            self.zData[:-1] = self.zData[1:]
            self.zData[-1] = self.z
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            self.ztime[:-1] = self.ztime[1:]
            self.ztime[-1] = self.zcurrentTime
            
            if self.framenumber % self.refresh:
                self.changedData.emit(self.time, self.xData, self.yData, self.ztime, self.zData)

        self.ptr += 1
    
    @pyqtSlot(bool, bool)
    def single_z_correction(self, feedback_val, initial):
        
        if initial:
        
            try:
                self.camera.stop_live_video()
                
            except: # TO DO: change for specific Error
                
                pass
            
            self.camera.start_live_video(framerate='20 Hz')
            self.camera._set_AOI(*self.roi_area)
            
            time.sleep(0.100)
        
        self.acquire_data()
        self.update_graph_data()
                
        if initial:
            
            self.setup_zfeedback()
            
        else:
        
            self.update_zfeedback(mode='discrete')#TODO change
        
        if self.save_data_state:
            
            self.ztime_array.append(self.zcurrentTime)
            self.z_array.append(self.focusSignal)
                    
        self.zIsDone.emit(True, self.target_z)
        
        
    @pyqtSlot(bool)
    def toggle_tracking(self, val):
        
        '''
        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of fiducial fluorescent beads. 
        Drift correction feedback loop is not automatically started.
        
        '''

        
        self.startTime = time.time()
        
        if val is True:
            
            self.reset()
            self.reset_data_arrays()
            
            self.tracking_value = True
            self.counter = 0
                    
        if val is False:
        
            self.tracking_value = False
    
    
    @pyqtSlot(bool)
    def toggle_ztracking(self, val):
        
        '''
        Connection: [frontend] ztrackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of confocal beam. 
        Drift correction feedback loop is not automatically started.
        
        '''

        
        self.startzTime = time.time()
        
        if val is True:

            self.reset()
            self.reset_data_arrays()
            print("turning z-tracking on")
            self.ztracking_value = True
            self.counter = 0
            
                    
        if val is False:
        
            self.ztracking_value = False
    
    
    def xcalibrate(self):

        #self.focusTimer.stop()
        time.sleep(0.100)
        self.toggle_tracking(True)
        #self.toggle_feedback(True)
        self.save_data_state = True
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, 
                                          self.zfeedback_active, 
                                          self.save_data_state, self.invertImage)
        
        self.reset()
        self.reset_data_arrays()
        
        currentXposition = self.pidevice.qPOS(1)[1]
        
        nsteps = 30
        Xmin = currentXposition - 0.05  # in µm
        Xmax = currentXposition + 0.05  # in µm
        Xrange = Xmax - Xmin  
        
        
        XData = np.arange(Xmin, Xmax+ Xrange/nsteps, Xrange/nsteps)
        calibData = np.zeros(XData.shape[0])
        if DEBUG:
            print(XData)
        XMoveTo(self.pidevice, Xmin)
        
        time.sleep(0.100)

        for i in range(XData.shape[0]):
            if DEBUG:
                print(i,XData[i])
            XMoveTo(self.pidevice, XData[i])
            time.sleep(0.70)
            self.update()
            time.sleep(0.05)
            calibData[i] =  self.y
            #added sleep 0.05 to ensure real point
        
        XMoveTo(self.pidevice, currentXposition)
        

        slope, intercept, r_value, p_value, std_err = stats.linregress(XData[2:] *1000 ,calibData[2:])
        print(slope, intercept)
        plt.plot(XData[2:] *1000 , intercept + slope*XData[2:]*1000, 'r-')
        plt.plot(XData[2:] *1000, calibData[2:], 'o')
#        plt.plot(XData *1000, calibData, 'o')
        plt.show()
        print("x= ", calibData)
        #self.pxSize = self.pxSize/slope 
        
        #TODO Export, and show, and linear  fit
        self.export_data("x_calib")
        self.reset_data_arrays()
        self.reset()
        time.sleep(0.200)
        
        #self.focusTimer.start(self.focusTime)
    
    
#    def xcalibrate(self):
#
#        #self.focusTimer.stop()
#        time.sleep(0.100)
#        self.toggle_tracking(True)
#        #self.toggle_feedback(True)
#        self.save_data_state = True
#        
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
#                                          self.feedback_active, 
#                                          self.zfeedback_active, 
#                                          self.save_data_state, self.invertImage)
#        
#        self.reset()
#        self.reset_data_arrays()
#        
#        currentXposition = self.pidevice.qPOS(2)[2]
#        
#        nsteps = 30
#        Xmin = currentXposition - 0.05  # in µm
#        Xmax = currentXposition + 0.05  # in µm
#        Xrange = Xmax - Xmin  
#        
#        
#        XData = np.arange(Xmin, Xmax+ Xrange/nsteps, Xrange/nsteps)
#        calibData = np.zeros(XData.shape[0])
#        if DEBUG:
#            print(XData)
#        YMoveTo(self.pidevice, Xmin)
#        
#        time.sleep(1.100)
#
#        for i in range(XData.shape[0]):
#            if DEBUG:
#                print(i,XData[i])
#            YMoveTo(self.pidevice, XData[i])
#            time.sleep(0.70)
#            self.update()
#            time.sleep(0.05)
#            calibData[i] =  self.x
#            #added sleep 0.05 to ensure real point
#        
#        YMoveTo(self.pidevice, currentXposition)
#        
#
#        slope, intercept, r_value, p_value, std_err = stats.linregress(XData[1:] *1000 ,calibData[1:])
#        print(slope, intercept)
#        plt.plot(XData[1:] *1000 , intercept + slope*XData[1:]*1000, 'r-')
#        plt.plot(XData[1:] *1000, calibData[1:], 'o')
#        plt.show()
#        print("x= ", calibData)
#        #self.pxSize = self.pxSize/slope 
#        self.pxSize = self.pxSize/abs(slope)
#        #TODO Export, and show, and linear  fit
#        self.export_data("x_calib")
#        self.reset_data_arrays()
#        self.reset()
#        time.sleep(0.200)
#        
        #self.focusTimer.start(self.focusTime)

            
        
    def calibrate(self):

        #self.focusTimer.stop()
        time.sleep(0.100)
        self.toggle_ztracking(True)        
        self.reset()
        self.reset_data_arrays()
        
        currentZposition = self.pidevice.qPOS(3)[3]
        self.save_data_state = True
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, 
                                          self.zfeedback_active, 
                                          self.save_data_state, self.invertImage)
                
        nsteps = 30
        zmin = currentZposition - 0.05  # in µm
        zmax = currentZposition + 0.05  # in µm
        zrange = zmax - zmin  
        
        
        zData = np.arange(zmin, zmax+ zrange/nsteps, zrange/nsteps)
        calibData = np.zeros(zData.shape[0])
        if DEBUG:
            print(zData)
        zMoveTo(self.pidevice, zmin)
        
        time.sleep(1)

        for i in range(zData.shape[0]):
            if DEBUG:
                print(i,zData[i])
            zMoveTo(self.pidevice, zData[i])
            time.sleep(0.7)
            self.update()
            #time.sleep(0.05)
            calibData[i] = self.z
            #added sleep 0.05 to ensure real point
        
        zMoveTo(self.pidevice, currentZposition)
        

        slope, intercept, r_value, p_value, std_err = stats.linregress(zData[2:] *1000 ,calibData[2:])
        print(slope, intercept)
        plt.plot(zData[2:] *1000 , intercept + slope*zData[2:]*1000, 'r-')
        plt.plot(zData[2:] *1000, calibData[2:], 'o')
        plt.show()
        
        self.zpxSize = self.zpxSize/slope 
        
        #TODO Export, and show, and linear  fit
        self.export_zdata("xyz_calib")
        self.reset_data_arrays()
        self.reset()
        time.sleep(0.200)
        
        #self.focusTimer.start(self.focusTime)
        
    def gaussian_fit(self):
        
        # set main reference frame
        roinumber = 0
        for roicoordinate in self.ROIcoordinates:
            
            xmin, xmax, ymin, ymax = roicoordinate
            xmin_nm, xmax_nm, ymin_nm, ymax_nm = roicoordinate * self.pxSize
            
            # select the data of the image corresponding to the ROI
    
            array = self.image[xmin:xmax, ymin:ymax]
            
            # set new reference frame
            
            xrange_nm = xmax_nm - xmin_nm
            yrange_nm = ymax_nm - ymin_nm
                 
            x_nm = np.arange(0, xrange_nm, self.pxSize)
            y_nm = np.arange(0, yrange_nm, self.pxSize)
            
            (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
            
            # find max 
            
            argmax = np.unravel_index(np.argmax(array, axis=None), array.shape)
            #print("argmax")
            #print(argmax)
            x_center_id = argmax[0]
            y_center_id = argmax[1]
            
            # define area around maximum
        
            xrange = 13 # in px
            yrange = 13 # in px
            
            xmin_id = int(x_center_id-xrange)
            xmax_id = int(x_center_id+xrange)
            
            ymin_id = int(y_center_id-yrange)
            ymax_id = int(y_center_id+yrange)
            
            array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                    
            xsubsize = 2 * xrange
            ysubsize = 2 * yrange
            
    #        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')
            
            x_sub_nm = np.arange(0, xsubsize) * self.pxSize
            y_sub_nm = np.arange(0, ysubsize) * self.pxSize
    
            [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
            
            # make initial guess for parameters
            
            bkg = np.min(array)
            A = np.max(array) - bkg
            σ = 50 # nm
            x0 = x_sub_nm[int(xsubsize/2)]
            y0 = y_sub_nm[int(ysubsize/2)]
            
            initial_guess_G = [A, x0, y0, σ, σ, bkg]
             
            poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                         array_sub.ravel(), p0=initial_guess_G)
            
            # retrieve results
    
            poptG = np.around(poptG, 2)
        
            A, x0, y0, σ_x, σ_y, bkg = poptG
            #if DEBUG:
            #    print(poptG)
            #TODO: changed! angle in correct
            self.currentx[roinumber] = x0 + Mx_nm[xmin_id, ymin_id]
            self.currenty[roinumber] = y0 + My_nm[xmin_id, ymin_id]
#            print(Mx_nm[xmin_id, ymin_id])
#            print(self.currentx[roinumber])
            roinumber = roinumber + 1
        
            
    def track(self):
        
        """ 
        Function to track fiducial markers (fluorescent beads) from the selected ROI.
        The position of the beads is calculated through a guassian fit. 
        If feedback_active = True it also corrects for drifts in xy
        If save_data_state = True it saves the xy data
        
        """
        

               
        if self.initial is True:
            
            self.initialx = np.copy(self.currentx) #copy so it is not linked in memory
            self.initialy = np.copy(self.currenty)
            
            self.initial = False
            
        self.x_beads = self.currentx - self.initialx 
        self.y_beads = self.currenty - self.initialy 
        
#        print(self.currentx )
#        print(self.initialx )
#        print(self.x_beads)
        self.x = np.mean(self.x_beads)
        self.y = np.mean(self.y_beads)
#        print(self.x)
        self.currentTime = time.time() - self.startTime
        
        if self.save_data_state:
            
            self.time_array[self.j] = self.currentTime
            self.x_array[self.j] = self.x_beads
            self.y_array[self.j] = self.y_beads
            
            self.j += 1
                        
            if self.j >= (self.buffersize - 10):    # TODO: -5 bad fix
                
                self.export_data()
                self.reset_data_arrays()
                
                
                if self.ztracking_value:
                    self.export_zdata()
                    self.reset_zdata_arrays()
                
                print(datetime.now(), '[xyz_tracking] Data array, longer than buffer size, data_array reset')
    
    def ztrack(self):

        self.currentz = self.zpxSize * self.massCenter[1] #changed form focusSignal
        #self.ztime = ptime.time() - self.startzTime
        #print(self.massCenter)
        if self.zinitial is True:
            
            self.initialz = self.currentz
            
            self.zinitial = False
            
        self.z = self.currentz - self.initialz
        self.zcurrentTime = time.time() - self.startzTime
        
        
        if self.save_data_state:
            #print(self.zj)
            self.ztime_array[self.zj] = self.zcurrentTime #TODO: wrong TIme?
            self.z_array[self.zj] = self.z
        
            
            self.zj += 1
                        
            if self.zj >= (self.buffersize - 10):    # TO DO: -5 bad fix
                
                self.export_zdata()
                self.reset_zdata_arrays()
                
                if self.tracking_value:
                    self.export_data()
                    self.reset_data_arrays()
                    
                    
                
                print(datetime.now(), '[xyz_tracking] Data array, longer than buffer size, data_array reset')
                
        
    
     
        
    def correct(self, mode='continous'):

        dx = 0
        dy = 0
        threshold = 10 # in nm
        far_threshold = 0.015 # in µm
        correct_factor = 0.6
        security_thr = 0.15 # in µm
        
        if np.abs(self.x) > threshold:
            
            dx = - (self.x)/1000
            
            if np.abs(dx) > far_threshold:
                
                dx = correct_factor * dx
            
             # conversion to µm

#                print('dx', dx)
            
        if np.abs(self.y) > threshold:
            
            dy =  - (self.y)/1000 # conversion to µm
            
            if np.abs(dy) > far_threshold:
                
                dy = correct_factor * dy
            
            
            
#                print('dy', dy)
    
        if dx > security_thr or dy > security_thr:
            
            print(datetime.now(), '[xyz_tracking] Correction movement larger than 200 nm, active correction turned OFF')
            self.toggle_feedback(False)
            
        else:
            
            # compensate for the mismatch between camera/piezo system of reference
            
            theta = np.radians(180)  #ehemals 90 # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            
            dy, dx = np.dot(R, np.asarray([dx, dy]))
            
            # add correction to piezo position

            
            currentXposition = self.pidevice.qPOS(1)[1]
            currentYposition = self.pidevice.qPOS(2)[2]
            
            targetXposition = currentXposition + dx  
            targetYposition = currentYposition + dy  
            
            if mode == 'continous':
            
                self.actuator_xy(targetXposition, targetYposition)
                
            if mode == 'discrete':
                
#                self.moveTo(targetXposition, targetYposition, 
#                            currentZposition, pixeltime=10)
                
                self.target_x = targetXposition
                self.target_y = targetYposition
        
        
    def zcorrect(self, mode='continous'):
        
        #dz = self.z * self.pxSize - self.setPoint
        dz = self.z - self.setPoint
        print("correcting dz = {}".format(dz))
#        print('dz', dz, ' nm')
        
        threshold = 13 # in nm
        far_threshold = 30 # in nm
        correct_factor = 0.6
        security_thr = 200 # in nm
        
        if np.abs(dz) > threshold:
            
            if np.abs(dz) < far_threshold:
                
                dz = correct_factor * dz
    
        if np.abs(dz) > security_thr:
            
            print(datetime.now(), '[focus] Correction movement larger than 200 nm, active correction turned OFF')
            #TODO: turnoff?
            
        else:
            
            self.target_z = self.target_z - dz/1000  # conversion to µm
  
            if mode is 'continous':
                    
                self.actuator_z(self.target_z)
                
            if mode is 'discrete':
                
                #pass  # it's enough to have saved the value self.target_z
                
                print(datetime.now(), '[focus] discrete correction to', self.target_z)
            
        
    #TODO: add single z correction
    @pyqtSlot(bool, bool)
    def single_xy_correction(self, feedback_val, initial): 
        
        """
        From: [psf] xySignal
        Description: Starts acquisition of the camera and makes one single xy
        track and, if feedback_val is True, corrects for the drift
        """
#        if DEBUG:
#            print(datetime.now(), '[xyz_tracking] Feedback {}'.format(feedback_val))
        
        if initial:
            self.toggle_feedback(True, mode='discrete')
            self.initial = initial
            print(datetime.now(), '[xyz_tracking] initial', initial)
                 


           
            self.camera.start_live_video(framerate='20 Hz')
            self.camera.master_gain = 1
            self.camera.exposure_time = "13ms"

        time.sleep(0.0500)

        self.image = self.camera.latest_frame()
        self.changedImage.emit(self.image)
            
        self.camera.stop_live_video()
        
        self.track()
        self.update_graph_data()
        self.correct(mode='discrete')
                
        target_x = np.round(self.target_x, 3)
        target_y = np.round(self.target_y, 3)
        
        print(datetime.now(), '[xyz_tracking] discrete correction to', 
              target_x, target_y)
    
        self.xyIsDone.emit(True, target_x, target_y)
        
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] single xy correction ended')  
        
        

        
    def actuator_xy(self, x_f, y_f):
        
#        print(datetime.now(), '[xyz_tracking] actuator x, y =', x_f, y_f)
        self.pidevice.MOV({1: x_f,2: y_f})
        time.sleep(0.05)
        for axis in self.pidevice.axes:
            position = self.pidevice.qPOS(axis)[axis]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
            print('current position of axis {} is {:.2f}'.format(axis, position))

    def set_moveTo_param(self, x_f, y_f, z_f ):
        
        self.pidevice.MOV({1: x_f,2: y_f, 3:z_f})
        time.sleep(0.05)
        for axis in self.pidevice.axes:
            position = self.pidevice.qPOS(axis)[axis]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
            print('current position of axis {} is {:.2f}'.format(axis, position))
        #self.pidevice.MOV('2', y_f)
        #self.pidevice.MOV('3', z_f)

    def moveTo(self, x_f, y_f, z_f, pixeltime=2000): 

        self.set_moveTo_param(x_f, y_f, z_f)
        
            
    def reset(self):
        #Bad HACK:
        self.x = 0
        self.y = 0
        self.z = 0
        self.currentTime = 0
        self.zcurrentTime = 0
        
        
        
        self.initial = True
        
        self.data = np.zeros(self.npoints)
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.zData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = time.time()
        self.j = 0  # iterator on the data array
        
        self.max_dev = 0
        self.mean = self.focusSignal
        self.std = 0
        self.n = 1
        
        #from z
        
        self.zinitial = True
        self.ztime = np.zeros(self.npoints)
        self.zptr = 0
        self.startzTime = time.time()
        self.zj = 0  # iterator on the data array
        
        
        

        self.changedData.emit(self.time, self.xData, self.yData, self.ztime, self.zData)
    
    
    
    def zreset(self):
        
        self.zinitial = True
        self.zData = np.zeros(self.npoints)
        self.ztime = np.zeros(self.npoints)
        self.ptr = 0
        self.startzTime = time.time()
        self.zj = 0  # iterator on the data array
        
        self.max_dev = 0
        self.mean = self.focusSignal
        self.std = 0
        self.n = 1
        
        #TODO: change emit with data?
        #self.changedzData.emit(self.ztime, self.zData)
    
    
    
    def reset_data_arrays(self):
                #TODO:adapter z
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        self.ztime_array = np.zeros(self.buffersize, dtype=np.float16)
        self.x_array = np.zeros((self.buffersize, self.numberofrois), dtype=np.float16)
        self.y_array = np.zeros((self.buffersize, self.numberofrois), dtype=np.float16)
        self.z_array = np.zeros(self.buffersize, dtype=np.float16)
        
        
    def reset_zdata_arrays(self):
                #TODO:adapter z
        self.ztime_array = np.zeros(self.buffersize, dtype=np.float16)
        self.z_array = np.zeros(self.buffersize, dtype=np.float16)
        
        zj = 0
        
        
        
    def export_data(self, inputfname = False):

        """
        Exports the x, y and t data into a .txt file
        """
        #TODO: make uniquename

        if inputfname != False:
            fname = inputfname
        else: 
            fname = self.filename
        filename = tools.getUniqueName(fname)    # TODO: make compatible with psf measurement and stand alone
        
        

        size = self.j #TODO:change further length

        for particle in range(self.numberofrois):
            
            savedData = np.zeros((3, size))
    
            savedData[0, :] = self.time_array[0:size]
            savedData[1, :] = self.x_array[0:size, particle]
            savedData[2, :] = self.y_array[0:size, particle]
            
            
            filenamereal = filename + '_xydata_particle_' + str(particle) + '.txt'        
            np.savetxt(filenamereal, savedData.T,  header='t (s), x (nm), y(nm)') # transpose for easier loading
        
        print(datetime.now(), '[xyz_tracking] xy data exported to', filenamereal)


    def export_zdata(self, inputfname = False):

        """
        Exports the z and t data into a .txt file
        """
        
        if inputfname != False:
            fname = inputfname
        else: 
            fname = self.filename
        filename = tools.getUniqueName(fname)    # TODO: make compatible with psf measurement and stand alone
        filenamereal = filename + '_zdata.txt'
        
    
        size = self.zj #TODO:change further length
        savedData = np.zeros((2, size))

        savedData[0, :] = self.ztime_array[0:self.zj]
        savedData[1, :] = self.z_array[0:self.zj]
        
        np.savetxt(filenamereal, savedData.T,  header='t (s), z (nm)') # transpose for easier loading
        
        
        
        print(datetime.now(), '[xyz_tracking] z data exported to', filenamereal)


   
    @pyqtSlot()    
    def get_stop_signal(self):
        
        """
        Connection: [psf] xyStopSignal
        Description: stops liveview, tracking, feedback if they where running to
        start the psf measurement with discrete xy - z corrections
        """
        
        self.toggle_feedback(False)
        self.toggle_zfeedback(False) 
        self.toggle_tracking(False)
        self.toggle_ztracking(False)
        
        self.reset()
        self.reset_data_arrays()
        
        self.zreset()
        self.reset_zdata_arrays()
        
        self.save_data_state = True  # TO DO: sync this with GUI checkboxes (Lantz typedfeat?)
        
        #self.liveview_stop()
    
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        Fals -> don't save
        
        '''
        
        self.save_data_state = val
        
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] save_data_state = {}'.format(val))
    

     
    @pyqtSlot(int, np.ndarray)
    def get_roi_info(self, N, coordinates_array):
        
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
        
        # TO DO: generalize to N ROIs
        self.numberofrois = N
        self.ROIcoordinates = coordinates_array.astype(int)
        
        self.currentx = np.zeros(self.numberofrois)
        self.currenty = np.zeros(self.numberofrois)
#        print(self.ROIcoordinates)
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] got ROI coordinates')
            

    @pyqtSlot(int, np.ndarray)
    def get_zroi_info(self, N, coordinates_array):
        
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
        
        # TO DO: generalize to N ROIs
        
        self.zROIcoordinates = coordinates_array.astype(int)
        print(self.zROIcoordinates)
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] got Z ROI coordinates')
        
        
        
        
    @pyqtSlot()    
    def get_lock_signal(self):
        #TODO: adapt?
        '''
        Connection: [minflux] xyzStartSignal
        Description: activates tracking and feedback
        
        '''
        
        
        self.getROISignal.emit()
        self.refresh = 5
        self.viewtimer.setInterval(self.measurementtime)
        self.toggle_zfeedback(False)
        self.toggle_feedback(False)
        self.reset()
        self.reset_data_arrays()
        
        self.zreset()
        self.reset_zdata_arrays()
        
        
        if self.ROIcoordinates != []:
            self.toggle_tracking(True)
            #self.toggle_feedback(True)
            self.toggle_ztracking(True)
    #        self.toggle_zfeedback(True)
            self.save_data_state = True
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, 
                                          self.zfeedback_active, 
                                          self.save_data_state, self.invertImage)
        
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] System xyz locked')

    @pyqtSlot(np.ndarray, np.ndarray) 
    def get_move_signal(self, r, r_rel):            
        
        self.toggle_feedback(False)
#        self.toggle_tracking(True)
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, 
                                          self.zfeedback_active, 
                                          self.save_data_state, self.invertImage)
        
        x_f, y_f = r

        self.actuator_xy(x_f, y_f)
         
        if DEBUG:
            print(datetime.now(), '[xyz_tracking] Moved to', r)
        
#        # Lock again
        
#        print(datetime.now(), '[xyz_tracking] initial x and y', self.initialx, self.initialy)
#        print(datetime.now(), '[xyz_tracking] dx, dy', r_rel)
##        self.initial = True # to lock at a new position, TO DO: fix relative position tracking
#        self.initialx = self.currentx - r_rel[0] * 1000 # r_rel to nm
#        self.initialy = self.currenty - r_rel[1] * 1000 # r_rel to nm
#        print(datetime.now(), '[xyz_tracking] initial x and y', self.initialx, self.initialy)
        
#        self.toggle_feedback(True) # TO DO: fix each position lock
        

#TODO: adapt get_tcspc_signal
#TODO: adapt get_scan_signal


        
    @pyqtSlot(str)    
    def get_end_measurement_signal(self, fname):
        
        '''
        From: [minflux] xyzEndSignal or [psf] endSignal
        Description: at the end of the measurement exports the xyz data

        '''
        
        
        self.viewtimer.setInterval(self.idletime)
        self.refresh = 2
        self.filename = fname
        self.export_data()
        self.export_zdata()
        self.toggle_feedback(False)
        self.toggle_zfeedback(False)
        self.toggle_tracking(False)
#        self.toggle_ztracking(False)
        self.updateGUIcheckboxSignal.emit(self.tracking_value, self.ztracking_value, 
                                          self.feedback_active, 
                                          self.zfeedback_active, 
                                          self.save_data_state, self.invertImage)
 
        
#        self.liveview_start()
#        time.sleep(0.4)
#        self.camera._set_AOI(*self.roi_area)
        
    def make_connection(self, frontend):
            
        frontend.liveviewSignal.connect(self.liveview)
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.zroiInfoSignal.connect(self.get_zroi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.exportzDataButton.clicked.connect(self.export_zdata)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.zreset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.clearDataButton.clicked.connect(self.reset_zdata_arrays)
        frontend.trackingBeadsBox.stateChanged.connect(lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked()))
        frontend.ztrackingBeadsBox.stateChanged.connect(lambda: self.toggle_ztracking(frontend.ztrackingBeadsBox.isChecked()))
        frontend.calibrationButton.clicked.connect(self.calibrate)
        frontend.xcalibrationButton.clicked.connect(self.xcalibrate)

        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.zfeedbackLoopBox.stateChanged.connect(lambda: self.toggle_zfeedback(frontend.zfeedbackLoopBox.isChecked()))
        
        #frontend.invertImageBox.stateChanged.connect(lambda: self.toggle_zfeedback(frontend.invertImageBox.isChecked()))
        #TODO what changes invert imagebox 
        # TO DO: clean-up checkbox create continous and discrete feedback loop
        
        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond 
        # (checked <-> active, not checked <-> inactive)
        
    def cam_stop(self):
        self._stop_event.set()

        
    @pyqtSlot()
    def stop(self):
        #self.focusTimer.stop()
        #self.viewTimer.stop() #TODO: viewTImer?
#        self.camera.close()
        self.cam_stop()
        
        """if standAlone is True:
            
            # Go back to 0 position
            x_0 = 2
            y_0 = 2
            z_0 = 5
            self.moveTo(x_0, y_0, z_0)"""
        print(datetime.now(), '[xyz_tracking] xyz_tracking stopped')
        
        # clean up aux files from NiceLib
        #TODO: CLEAN UP
        #os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\lextab.py')
        #os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\yacctab.py')
            

if __name__ == '__main__':

    app = QtGui.QApplication([])
#    app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    print(datetime.now(), '[xyz_tracking] Focus lock module running in stand-alone mode')
    standAlone = True
    # initialize devices
    
    #cam = uc480.UC480_Camera()
#    paramsets = list_instruments()
#    print(paramsets)
#    cam = uc480.UC480_Camera(paramsets[1])
    
    sdk =  TLCameraSDK()
    camera_list = sdk.discover_available_cameras()
    cam = sdk.open_camera(camera_list[0])
    

    
    CONTROLLERNAME = 'E-727'
    STAGES = ('P-733.3CD',)
    
    
    pidevice= GCSDevice(CONTROLLERNAME)
   
    """pidevice.ConnectUSB(serialnum='0118038033')
    if pidevice.HasqVER():
        print('version info:\n{}'.format(pidevice.qVER().strip()))
    
    pidevice.send("SVO 1 1")
    pidevice.send("SVO 2 1")
    pidevice.send("SVO 3 1")
    #scan.setupDevice(pidevice)
    """
    gui = Frontend()
    worker = Backend(cam, pidevice)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
        
    
        
        #worker.adw.Set_FPar(70, pos_zero)
        #worker.adw.Set_FPar(71, pos_zero)
        #worker.adw.Set_FPar(72, pos_zero)
    
        #worker.moveTo(10, 10, 10) # in µm
    
    time.sleep(0.200)
        
        #worker.piezoXposition = 10.0 # in µm
        #worker.piezoYposition = 10.0 # in µm
        #worker.piezoZposition = 10.0 # in µm

    gui.setWindowTitle('xyz drift correction')
    gui.show()
    app.exec_()
        