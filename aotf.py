# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:33:06 2021

@author: MINFLUX
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

import drivers.AOTF_driver as AOTF_driver


class Backend(QtCore.QObject):
    
#    tcspcPrepareSignal = pyqtSignal(str, int, int)
#    tcspcStartSignal = pyqtSignal()
#    
#    xyzStartSignal = pyqtSignal()
#    xyzEndSignal = pyqtSignal(str)
#    
#    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    

    
    
    
    def __init__(self, aotf, *args, **kwargs):
        
        
        
        super().__init__(*args, **kwargs)
        
        
        
        self.aotf = aotf
        
        self.freq00 = 118.0 # 532nm
        self.amp00 = 5000
        
        self.freq01 = 115.9 # 532nm
        self.amp01 = 5000
        
        #self.freq1 = 92.931 # 635 nm
#        self.freq10 = 91.6 # 640 nm
#        self.amp10 = 5000
        
        self.freq10 = 92.0 # 640 nm #IRF 89
        self.amp10 = 5000       
        
        self.freqIRF = 89.0 # 640 nm #IRF 89
        self.ampIRF = 5000       
        
        self.freqIRFgreen = 112.0 # 640 nm #IRF 89
        self.ampIRF = 5000       
#        self.freq11 = 93.2 # 640 nm
#        self.amp11 = 5000
        self.alex = False
        self.sequential = False
        
        self.alexTimer = QtCore.QTimer()
        self.sequentialTimer = QtCore.QTimer()
        
        self.init_AOTF()
        
        
        
        
    def init_AOTF(self):
        if not self.aotf.getStatus():
            exit()
        
        print(self.aotf._sendCmd("BoardID ID"))
        print(self.aotf._sendCmd("dds f 0 "+ str(self.freq00)))
        print(self.aotf._sendCmd("dds f 1 "+ str(self.freq01)))
        
        print(self.aotf._sendCmd("dds f 2 "+ str(self.freq10)))
        
        print(self.aotf._sendCmd("dds a 4 0") )
        
        print(self.aotf._sendCmd("dds f 4 "+ str(self.freqIRF)))
        print(self.aotf._sendCmd("dds a 5 0") )
        print(self.aotf._sendCmd("dds f 5 "+ str(self.freqIRFgreen)))
#        print(self.aotf._sendCmd("dds f 3 "+ str(self.freq11)))
        
        
        
        
        
    #opnly works with one wavelength!!!!
    @pyqtSlot(bool)
    def set_greennew(self, val):
        if val:
            self.aotf._sendCmdnew("dds a 0 "+str(self.amp00))
            self.aotf._sendCmdnew("dds a 1 "+str(self.amp01))
        else:
            self.aotf._sendCmdnew("dds a 0 0")
            self.aotf._sendCmdnew("dds a 1 0")
            
    @pyqtSlot(bool)        
    def set_rednew(self, val):
        
        
        if val:
            self.aotf._sendCmdnew("dds a 2 "+str(self.amp10))  
            self.aotf._sendCmdnew("dds a 3 "+str(self.amp11)) 
        else:
            self.aotf._sendCmdnew("dds a 2 0")
            self.aotf._sendCmdnew("dds a 3 0")
            
    @pyqtSlot(bool)
    def IRFred(self, val):
        if val:
#            self.aotf._sendCmd("dds a 0 "+str(self.amp00))
            self.aotf._sendCmd("dds a 4 "+str(self.ampIRF))
        else:
#            self.aotf._sendCmd("dds a 0 0")
            self.aotf._sendCmd("dds a 4 0") 
            
    @pyqtSlot(bool)
    def IRFgreen(self, val):
        if val:
#            self.aotf._sendCmd("dds a 0 "+str(self.amp00))
            self.aotf._sendCmd("dds a 5 "+str(self.ampIRF))
        else:
#            self.aotf._sendCmd("dds a 0 0")
            self.aotf._sendCmd("dds a 5 0") 
                 
    @pyqtSlot(bool)
    def set_green(self, val):
        if val:
            self.aotf._sendCmd("dds a 0 "+str(self.amp00))
            self.aotf._sendCmd("dds a 1 "+str(self.amp01))
        else:
            self.aotf._sendCmd("dds a 0 0")
            self.aotf._sendCmd("dds a 1 0")
            
    @pyqtSlot(bool)        
    def set_red_dual(self, val):
        
        
        if val:
            self.aotf._sendCmd("dds a 2 "+str(self.amp10))  
            self.aotf._sendCmd("dds a 3 "+str(self.amp11)) 
        else:
            self.aotf._sendCmd("dds a 2 0")
            self.aotf._sendCmd("dds a 3 0")
            
            
    @pyqtSlot(bool)        
    def set_red(self, val):
        
        
        if val:
            self.aotf._sendCmd("dds a 2 "+str(self.amp10))  
#            self.aotf._sendCmd("dds a 3 "+str(self.amp11)) 
        else:
            self.aotf._sendCmd("dds a 2 0")
#            self.aotf._sendCmd("dds a 3 0")
                     

    
    @pyqtSlot(str, float, str)
    def get_init_Signal(self, aotf_pattern, dt, start_color):
        print("get aotf signal", aotf_pattern, dt, start_color)
        self.updateTime = int(dt * 1000)
        self.start_color = start_color
        self.aotf_pattern = aotf_pattern
        self.IRFred(False)
        self.IRFgreen(False)
        
        
        if self.start_color == "green":
            self.set_green(True)
            self.set_red(False)
                
        elif self.start_color == "red":
            self.set_green(False)
            self.set_red(True)
        
        
        
        if self.aotf_pattern == "green":
            self.alex = False
            self.sequential = False
        
        if self.aotf_pattern == "red":
            self.alex = False
            self.sequential = False
            
        if self.aotf_pattern == "alex":
            self.alexTimer.setInterval(self.updateTime)
            self.alex = True
            self.sequential = False
            self.alexCounter = 0
            
        if self.aotf_pattern == "sequential":
            self.sequentialTimer.setInterval(self.updateTime)
            self.alex = False
            self.sequential = True   
            
            
        if self.aotf_pattern == "IRFred":
            self.set_green(False)
            self.set_red(False)
            self.alex = False
            self.sequential = False   
            self.IRFred(True)
            
        
        if self.aotf_pattern == "IRFgreen":
            self.set_green(False)
            self.set_red(False)
            self.alex = False
            self.sequential = False     
            self.IRFgreen(True)
            #T
            
            
            
            
        
    @pyqtSlot() 
    def start_timer(self):
        
        if self.alex:
            self.alexTimer.start()
        elif self.sequential:
            self.sequentialTimer.start()
    
    
    
    def do_alex(self):
        
        if self.alexCounter %2 == 0:
            if self.start_color == "green":
                self.set_red(False)
                self.set_green(True)
                
                
            elif self.start_color == "red":
                self.set_green(False)
                self.set_red(True)
                
                
                
        elif self.alexCounter %2 == 1:
            if self.start_color == "green":
                self.set_green(False)
                self.set_red(True)
                
            elif self.start_color == "red":
                self.set_red(False)
                self.set_green(True)
                
    
        self.alexCounter += 1
    

    
    def do_sequential(self):
        
        
        self.sequentialTimer.stop()
        
        if self.start_color == "green":
            self.set_green(False)
            self.set_red(True)
            
        elif self.start_color == "red":
            self.set_red(False)
            self.set_green(True)
            
    
    @pyqtSlot()
    def stopALEX(self):
        self.alexTimer.stop()
    
    @pyqtSlot(bool)
    def setall(self, val):
        if val:
            self.set_green(True)
            self.set_red(True)
        else: 
            self.set_green(False)
            self.set_red(False)
        

        
    @pyqtSlot()
    def setgreen(self):
        self.set_green(True)
        self.set_red(False)
        
    @pyqtSlot()
    def setred(self):
        self.set_green(False)
        self.set_red(True)
        
    def stop(self):
        self.aotf.shutDown()
    
if __name__ == '__main__':
    
    aotf = AOTF_driver.AOTF64Bit(python32_exe = "C:/Users/MINFLUX/AppData/Local/Programs/Python/Python36-32/python")
    
    if not aotf.getStatus():
        exit()

    print(aotf._sendCmd("BoardID ID"))
    aotf.shutDown()