# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018


PyFlux Masterfile

@author:  Jonas Zaehringer
original template: Luciano A. Masullo


"""


def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    input("Press key to exit.")
    sys.exit(-1)

import sys
sys.excepthook = show_exception_and_exit


import numpy as np
import time
import os
from datetime import date, datetime

from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import qdarkstyle

from instrumental.drivers.cameras import uc480
from instrumental import instrument, list_instruments

import queue

import lantz.drivers.andor.ccd as ccd
import drivers.hydraharp as hydraharp

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog

#import drivers
from pipython import GCSDevice, pitools
import drivers.owis_ps10.ps10 as owis



import scan_FLIM_v2 as scan
import tcspc as tcspc
import xyz_tracking_multitarget_newcam_mp as xyz_tracking_multitarget_newcam

import measurements.minflux as minflux
import measurements.psf as psf
import measurements.PickAndDestroy as PickAndDestroy


import tools.tools as tools

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from thorlabs_windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

    
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK


import multiprocessing as mp

π = np.pi


class Frontend(QtGui.QMainWindow):
    
    closeSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyFLUX')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Actions in menubar

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Measurement')
        
        self.psfWidget = psf.Frontend()
        self.minfluxWidget = minflux.Frontend()
        self.PickAndDestroyWidget = PickAndDestroy.Frontend()        

        self.psfMeasAction = QtGui.QAction('PSF measurement', self)
        self.psfMeasAction.setStatusTip('Routine to measure one MINFLUX PSF')
        fileMenu.addAction(self.psfMeasAction)
        
        self.psfMeasAction.triggered.connect(self.psf_measurement)
    
        self.minfluxMeasAction = QtGui.QAction('MINFLUX measurement', self)
        self.minfluxMeasAction.setStatusTip('Routine to perform a tcspc-MINFLUX measurement')
        fileMenu.addAction(self.minfluxMeasAction)
        
        self.minfluxMeasAction.triggered.connect(self.minflux_measurement)
        
        
        
        
        self.PickAndDestroyMeasAction = QtGui.QAction('Pick and Destroy measurement', self)
        self.PickAndDestroyMeasAction.setStatusTip('Routine to perform multiple tcspc-MINFLUX measurement')
        fileMenu.addAction(self.PickAndDestroyMeasAction)
        
        self.PickAndDestroyMeasAction.triggered.connect(self.PickAndDestroy_measurement)
        
        

        # GUI layout

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        # Dock Area

        dockArea = DockArea()
        grid.addWidget(dockArea, 0, 0)

        ## scanner

        scanDock = Dock('scan', size=(1, 1))

        self.scanWidget = scan.Frontend()

        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock, 'left')

        ## tcspc

        tcspcDock = Dock("Time-correlated single-photon counting")

        self.tcspcWidget = tcspc.Frontend()

        tcspcDock.addWidget(self.tcspcWidget)
        dockArea.addDock(tcspcDock, 'bottom', scanDock)

        ## xyz tracking

        xyzDock = Dock("xyz drift control")

        self.xyzWidget = xyz_tracking_multitarget_newcam.Frontend()

        xyzDock.addWidget(self.xyzWidget)
        dockArea.addDock(xyzDock, 'right')
#        dockArea.addDock(xyDock, 'top')

        # sizes to fit my screen properly

        self.scanWidget.setMinimumSize(1000, 550)
        self.xyzWidget.setMinimumSize(800, 300)
        self.move(1, 1)
        
    def make_connection(self, backend):
        
        #backend.zWorker.make_connection(self.focusWidget)
        backend.scanWorker.make_connection(self.scanWidget)
        backend.tcspcWorker.make_connection(self.tcspcWidget)
        backend.xyzWorker.make_connection(self.xyzWidget)
        
        backend.minfluxWorker.make_connection(self.minfluxWidget)
        backend.psfWorker.make_connection(self.psfWidget)
        backend.PickAndDestroyWorker.make_connection(self.PickAndDestroyWidget)

    def psf_measurement(self):

        self.psfWidget.show()
        
    def minflux_measurement(self):
        
        self.minfluxWidget.show()
        self.minfluxWidget.emit_filename()

    def PickAndDestroy_measurement(self):
        
        self.PickAndDestroyWidget.show()
        self.PickAndDestroyWidget.emit_filename()

    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()

        super().closeEvent(*args, **kwargs)



        
class Backend(QtCore.QObject):
    
    askROIcenterSignal = pyqtSignal()
    moveToSignal = pyqtSignal(np.ndarray)
    tcspcStartSignal = pyqtSignal(str, int, int)
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyMoveAndLockSignal = pyqtSignal(np.ndarray)
    
    def __init__(self, pidevice, hh, owisps10, cam, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        
#        print("I'm running on CPU #%s" % mp.current_process().name)
        scanTCSPCdata = queue.Queue()
        
        self.scanWorker = scan.Backend(pidevice, owisps10, scanTCSPCdata)
        #self.zWorker = focus.Backend(scmos, pidevice)
#        pool = mp.Pool(2, initializer=init, initargs = (1,1))
#        print(pool)
        
        self.tcspcWorker = tcspc.Backend(hh, scanTCSPCdata)
        self.xyzWorker = xyz_tracking_multitarget_newcam.Backend(cam, pidevice)
        
        self.minfluxWorker = minflux.Backend(pidevice)
        self.psfWorker = psf.Backend()
        
        self.PickAndDestroyWorker = PickAndDestroy.Backend()
            
    def setup_minflux_connections(self):
        
        self.scanWorker.ROIcenterSignal.connect(self.minfluxWorker.get_ROI_center)
        
        self.minfluxWorker.tcspcPrepareSignal.connect(self.tcspcWorker.prepare_minflux)
        self.minfluxWorker.tcspcStartSignal.connect(self.tcspcWorker.measure_minflux)
        
        self.minfluxWorker.xyzStartSignal.connect(self.xyzWorker.get_lock_signal)
#        self.minfluxWorker.xyzStartSignal.connect(self.zWorker.get_lock_signal)
        
        self.minfluxWorker.moveToSignal.connect(self.xyzWorker.get_move_signal)
        
        self.tcspcWorker.tcspcDoneSignal.connect(self.minfluxWorker.get_tcspc_done_signal)
        self.minfluxWorker.setODSignal.connect(self.scanWorker.setODtcspc)        
        self.minfluxWorker.xyzEndSignal.connect(self.xyzWorker.get_end_measurement_signal)
        #self.minfluxWorker.xyzEndSignal.connect(self.zWorker.get_end_measurement_signal)
        
        
    def setup_pickanddestroy_connections(self):
        self.scanWorker.found_pick_PickandDestroy.connect(self.PickAndDestroyWorker.Destroy_PickandDestroy)
        self.PickAndDestroyWorker.scan_PickAndDestroySignal.connect(self.scanWorker.find_PickandDestroy)
        self.PickAndDestroyWorker.MINFLUXDestroy_PickAndDestroySignal.connect(self.minfluxWorker.Destroy_PickAndDestroy)
        
        self.PickAndDestroyWorker.requestScanImagetoScanSignal.connect(self.scanWorker.sendImagetoPD)
        self.scanWorker.imageSignalPD.connect(self.PickAndDestroyWorker.getScanImage)
        
        
        
        self.tcspcWorker.tcspcPickAndDestroySignal.connect(self.PickAndDestroyWorker.loop_PickandDestroy)
        
        
        
    def setup_psf_connections(self):
        
        self.psfWorker.scanSignal.connect(self.scanWorker.get_scan_signal)
        self.psfWorker.xySignal.connect(self.xyzWorker.single_xy_correction) #TODO Check xySignal
        self.psfWorker.zSignal.connect(self.xyzWorker.single_z_correction) #TODO: check if wrong
        self.psfWorker.xyStopSignal.connect(self.xyzWorker.get_stop_signal)
        self.psfWorker.zStopSignal.connect(self.xyzWorker.get_stop_signal)
        self.psfWorker.moveToInitialSignal.connect(self.scanWorker.get_moveTo_initial_signal)
        
        self.psfWorker.endSignal.connect(self.xyzWorker.get_end_measurement_signal)#TODO: check if bad
        #self.psfWorker.endSignal.connect(self.xyzWorker.get_end_measurement_signal) 
        
        self.scanWorker.frameIsDone.connect(self.psfWorker.get_scan_is_done)
        self.xyzWorker.xyIsDone.connect(self.psfWorker.get_xy_is_done)
        self.xyzWorker.zIsDone.connect(self.psfWorker.get_z_is_done)
    
    
    
    def setup_scan_connections(self):
        
        
        self.scanWorker.tcspcPrepareScanSignal.connect(self.tcspcWorker.prepare_scan)
#        self.scanWorker.tcspcPrepareHistoScanSignal.connect(self.tcspcWorker.prepare_hh_histo)
#        self.scanWorker.tcspcMeasureScanSignal.connect(self.tcspcWorker.measure_scan)
        self.scanWorker.tcspcMeasureFastScanSignal.connect(self.tcspcWorker.measure_fastscan)
        self.scanWorker.tcspcCloseConnection.connect(self.tcspcWorker.closeconnection)
        self.scanWorker.tcspcMeasureScanPointSignal.connect(self.tcspcWorker.measure_fastscan_Point)
        
        
        
    def setup_tcspc_connections(self):
        
        self.tcspcWorker.toggleShutterSignal.connect(self.scanWorker.toggle_shutter)
        self.tcspcWorker.toggleShutterSignalMINFLUX.connect(self.scanWorker.toggle_shutter_all)
        self.tcspcWorker.setODSignal.connect(self.scanWorker.setODtcspc)
                
    
    
    def make_connection(self, frontend):
        
        #frontend.focusWidget.make_connection(self.zWorker)
        frontend.scanWidget.make_connection(self.scanWorker)
        frontend.tcspcWidget.make_connection(self.tcspcWorker)
        frontend.xyzWidget.make_connection(self.xyzWorker)
    
    
        frontend.PickAndDestroyWidget.make_connection(self.PickAndDestroyWorker)
        frontend.minfluxWidget.make_connection(self.minfluxWorker)
        frontend.psfWidget.make_connection(self.psfWorker)
    
        self.setup_minflux_connections()
        self.setup_psf_connections()
        self.setup_scan_connections()
        self.setup_tcspc_connections()
        self.setup_pickanddestroy_connections()
        frontend.scanWidget.paramSignal.connect(self.psfWorker.get_scan_parameters)
        # TO DO: write this in a cleaner way, i. e. not in this section, not using frontend
        
        frontend.closeSignal.connect(self.stop)
        
    def stop(self):
        
        self.scanWorker.stop()
        #self.zWorker.stop()
        self.tcspcWorker.stop()
        self.xyzWorker.stop()

if __name__ == '__main__':

    app = QtGui.QApplication([])
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    dark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setStyleSheet(dark_stylesheet)
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    icon_path = r'pics/icon.jpg'
    app.setWindowIcon(QtGui.QIcon(icon_path))
    gui = Frontend()
    
    
    sdk =  TLCameraSDK()
    camera_list = sdk.discover_available_cameras()
    cam = sdk.open_camera(camera_list[0])
    
    
    hh = hydraharp.HydraHarp400()
    
    CONTROLLERNAME = 'E-727'
    STAGES = ('P-733.3CD',)
    REFMODES = None
    #init stage
    
    TCSPCdone = False

    owisps10 = owis.PS10("PS10-1_2.owi")
    
    
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectUSB(serialnum='0118038033')
        if pidevice.HasqVER():
            print('version info:\n{}'.format(pidevice.qVER().strip()))
        
        #initialize Servos #TODO: serach fo rmore inits autozero?
        pidevice.send("SVO 1 1")
        pidevice.send("SVO 2 1")
        pidevice.send("SVO 3 1")


        #scan.setupDevice(pidevice)

        worker = Backend(pidevice, hh, owisps10, cam)

        gui.make_connection(worker)
        worker.make_connection(gui)
    
        # initial parameters
    
        gui.scanWidget.emit_param()
        worker.scanWorker.emit_param()
        
        
        #TODO: check if needed 06_02_2020 git overwritten
        #gui.minfluxWidget.emit_param()
        #gui.minfluxWidget.emit_param_to_frontend 
        #gui.minfluxWidget.emit_param_to_backend()
        #worker.minfluxWorker.emit_param_to_frontend()
    
        gui.psfWidget.emit_param()
    
#    # GUI thread
#    
#    guiThread = QtCore.QThread()
#    gui.moveToThread(guiThread)
#    
#    guiThread.start()
    
    # psf thread
    
#    psfGUIThread = QtCore.QThread()
#    gui.psfWidget.moveToThread(psfGUIThread)
#    
#    psfGUIThread.start()
    
    # focus thread

    
        """
        focusThread = QtCore.QThread()
        worker.zWorker.moveToThread(focusThread)
        worker.zWorker.focusTimer.moveToThread(focusThread)
        worker.zWorker.focusTimer.timeout.connect(worker.zWorker.update)

        focusThread.start()
        """
        # focus GUI thread
    
#    focusGUIThread = QtCore.QThread()
#    gui.focusWidget.moveToThread(focusGUIThread)
#    
#    focusGUIThread.start()
    
#    # xy worker thread
        xyzThread = QtCore.QThread()
        worker.xyzWorker.moveToThread(xyzThread)
        worker.xyzWorker.viewtimer.moveToThread(xyzThread)
#    worker.xyWorker.viewtimer.timeout.connect(worker.xyWorker.update_view)
#    
        xyzThread.start()
    
    # xy GUI thread
    
#    xyGUIThread = QtCore.QThread()
#    gui.xyWidget.moveToThread(xyGUIThread)R
#    
#    xyGUIThread.start()

        # tcspc thread
    
        tcspcWorkerThread = QtCore.QThread()
        worker.tcspcWorker.moveToThread(tcspcWorkerThread)
        worker.tcspcWorker.syncTimer.moveToThread(tcspcWorkerThread)
        worker.tcspcWorker.syncTimer.timeout.connect(worker.tcspcWorker.update_view)
        
        tcspcWorkerThread.start()
    
        # scan thread
    
        scanThread = QtCore.QThread()
    
        worker.scanWorker.moveToThread(scanThread)
        worker.scanWorker.viewtimer.moveToThread(scanThread)
        worker.scanWorker.viewtimer.timeout.connect(worker.scanWorker.update_view)

        scanThread.start()
    
        # minflux worker thread
    
        minfluxThread = QtCore.QThread()
        worker.minfluxWorker.moveToThread(minfluxThread)
    
        minfluxThread.start()
    
    
    
    
        PickAndDestroyThread = QtCore.QThread()
        worker.PickAndDestroyWorker.moveToThread(PickAndDestroyThread)
    
        PickAndDestroyThread.start()
        # psf worker thread
    
#    psfThread = QtCore.QThread()
#    worker.psfWorker.moveToThread(psfThread)
#    worker.psfWorker.measTimer.moveToThread(psfThread)
#    worker.psfWorker.measTimer.timeout.connect(worker.psfWorker.measurement_loop)
#
#    psfThread.start()
    
    # minflux measurement connections
    
        gui.show()
        app.exec_()
