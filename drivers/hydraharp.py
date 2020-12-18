# -*- coding: utf-8 -*-
"""
Created on  11/06/19

@author: Jonas Zaehringer
"""

import time
import ctypes as ct
from ctypes import byref, POINTER
from lantz import LibraryDriver
from lantz import Driver, Feat, Action
import time
import sys
from datetime import datetime
import numpy as np
import os
import multiprocessing as mp

# From hhdefin.h
LIB_VERSION = "3.0"
MAXDEVNUM = 8
MODE_T2 = 2
MODE_T3 = 3
MAXLENCODE = 6
HHMAXINPCHAN = 8
MAXHISTLEN = 65536
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0001
FLAG_FIFOFULL = 0x0002
DEV_NUM = 0     # device number, works for only 1 device

class HydraHarp400(LibraryDriver):
    
    LIBRARY_NAME  = "hhlib64.dll"
    
    def __init__(self, *args, **kwargs):
        super().__init__(library_name="hhlib64.dll" ,*args, **kwargs)
        
        
        # Measurement parameters, these are hardcoded since this is just a demo
        self.mode = MODE_T3 # set T2 or T3 here, observe suitable Syncdivider and Range!
        self.binning = 0 # you can change this, meaningful only in T3 mode
        self.offset = 0 # you can change this, meaningful only in T3 mode
        self.tacq = 1000 # Measurement time in millisec, you can change this
        self.syncDiv = 1 # you can change this, observe mode! READ MANUAL!
        self.syncCFDZeroCross = 10 # you can change this (in mV)
        self.syncCFDLevel = 50 # you can change this (in mV)
        self.syncChannelOffset = -13000 # you can change this (in ps, like a cable delay)
        self.inputCFDZeroCross = 10 # you can change this (in mV)
        self.inputCFDLevel = 50 # you can change this (in mV)
        self.inputChannelOffset = -27750 # you can change this (in ps, like a cable delay)
        
        self.maxRes = 1 # max res of HydraHarp 400 in ps
        
        # Variables to store information read from DLLs copied of demo
        self.buffer = (ct.c_uint * TTREADMAX)()
        self.dev = []
        self.libVersion = ct.create_string_buffer(b"", 8)
        self.hwSerial = ct.create_string_buffer(b"", 8)
        self.hwPartno = ct.create_string_buffer(b"", 8)
        self.hwVersion = ct.create_string_buffer(b"", 8)
        self.hwModel = ct.create_string_buffer(b"", 16)
        self.errorString = ct.create_string_buffer(b"", 40)
        self.numChannels = ct.c_int()
        self.res = ct.c_double() 
        self.syncRate = ct.c_int()
        #self.countRate = ct.c_int()
        self.countRate0 = ct.c_int()
        self.countRate1 = ct.c_int()
        self.countRate2 = ct.c_int()
        self.countRate3 = ct.c_int()
        self.flags = ct.c_int()
        self.nRecords = ct.c_int()
        self.ctcstatus = ct.c_int()
        self.warnings = ct.c_int()
        self.warningstext = ct.create_string_buffer(b"", 16384)
        
        self.histcounts = [(ct.c_uint * MAXHISTLEN)() for i in range(0, HHMAXINPCHAN)]
        self.histLen = ct.c_int()
    def getLibraryVersion(self):
        
        self.lib.HH_GetLibraryVersion(self.libVersion)
        
        
        return self.libVersion.value.decode("utf-8")
        
        
    @Action()
    def open(self):
        print("opening HydraHarp")
        print(DEV_NUM)
        print(self.hwSerial)
        retcode = self.lib.HH_OpenDevice(ct.c_int(DEV_NUM), self.hwSerial)
        if retcode == 0:
            print("  %1d     S/N %s" % (DEV_NUM, 
                                        self.hwSerial.value.decode("utf-8")))

        else:
            if retcode == -1: # ERROR_DEVICE_OPEN_FAIL
                print("  %1d     no device" % DEV_NUM)
            else:
                self.lib.HH_GetErrorString(self.errorString, 
                                           ct.c_int(retcode))
                
                print("  %1d     %s" % (DEV_NUM, 
                                        self.errorString.value.decode("utf8")))
                                        
                                        
                                        
    def getHardwareInfo(self):
        
        self.lib.HH_GetHardwareInfo(DEV_NUM, self.hwModel, self.hwPartno, 
                                    self.hwVersion)
        
        return [self.hwModel.value.decode("utf-8"),
               self.hwPartno.value.decode("utf-8"),
               self.hwVersion.value.decode("utf-8")]
               

    def setup(self):
        
        self.lib.HH_Calibrate(ct.c_int(DEV_NUM))
        
        self.lib.HH_SetSyncDiv(ct.c_int(DEV_NUM), 
                               ct.c_int(self.syncDiv))
                               
                               

        for i in range(0, 2):
            self.lib.HH_SetInputCFD(ct.c_int(DEV_NUM), ct.c_int(i), ct.c_int(self.inputCFDLevel),\
                                    ct.c_int(self.inputCFDZeroCross))

            self.lib.HH_SetInputChannelOffset(ct.c_int(DEV_NUM), ct.c_int(i),\
                                            ct.c_int(self.inputChannelOffset))

        """# Meaningful only in T3 mode
        if self.mode == MODE_T3:
            self.lib.HH_SetBinning(ct.c_int(DEV_NUM), ct.c_int(self.binning))
            self.lib.HH_SetOffset(ct.c_int(DEV_NUM), ct.c_int(self.offset))
            self.lib.HH_GetResolution(ct.c_int(DEV_NUM), byref(self.res))
            print("Resolution is %1.1lfps" % self.res.value)"""
        
        time.sleep(0.2)
        
    
    def histosetup(self):
        
        self.lib.HH_SetHistoLen(ct.c_int(DEV_NUM), ct.c_int(MAXLENCODE), byref(self.histLen))
        self.lib.HH_SetBinning(ct.c_int(DEV_NUM), ct.c_int(self.binning))
        self.lib.HH_SetOffset(ct.c_int(DEV_NUM), ct.c_int(self.offset))
        
        
    
    
    
    @Feat 
    def binning(self):
        
        return self.binningValue
        
    @binning.setter 
    def binning(self, value):
        
        self.lib.HH_SetBinning(ct.c_int(DEV_NUM), 
                               ct.c_int(value))
        self.binningValue = value
        
        
        
    @Feat    
    def offset(self):
        
        return self.offsetValue
        
    @offset.setter
    def offset(self,value):
        
        self.lib.HH_SetOffset(ct.c_int(DEV_NUM), ct.c_int(value))
        self.offsetValue = value
        
    @Feat
    def resolution(self):
        
        self.lib.HH_GetResolution(ct.c_int(DEV_NUM), 
                                  byref(self.res))
        
        return self.res.value
    
    @resolution.setter
    def resolution(self, value):
        
        # calculation: resolution = maxRes * 2**binning
        #print("resolution value")
        #print(value)
        #print(type(value))
        if  value > 0:
            self.binning = int((np.log(value)-np.log(self.maxRes))/np.log(2))
        else:
            print("[HydraHarp] Resolution value needs to be bigger than 0")
            pass
        
    def countrate(self, channel):
        
        if channel == 0:
            
            self.lib.HH_GetCountRate(ct.c_int(DEV_NUM), ct.c_int(0), 
                                     byref(self.countRate0))
            
            value = self.countRate0.value
            
        if channel == 1:
            
            self.lib.HH_GetCountRate(ct.c_int(DEV_NUM), ct.c_int(1), 
                                     byref(self.countRate1))
            
            value = self.countRate1.value
            
        if channel == 2:
            
            self.lib.HH_GetCountRate(ct.c_int(DEV_NUM), ct.c_int(2), 
                                     byref(self.countRate1))
            
            value = self.countRate1.value
            
        if channel == 3:
            
            self.lib.HH_GetCountRate(ct.c_int(DEV_NUM), ct.c_int(3), 
                                     byref(self.countRate1))
            
            value = self.countRate1.value
        
        return value
        
        

    @Feat
    def syncDivider(self):
        
        return self.syncDiv
      
    @syncDivider.setter
    def syncDivider(self, value):
        
        self.lib.HH_SetSyncDiv(ct.c_int(DEV_NUM), ct.c_int(value))
        self.syncDiv = value
        
        
        
    def getSyncRate(self):
               
        self.lib.HH_GetSyncRate(ct.c_int(DEV_NUM), byref(self.syncRate))
        #print("\nSyncrate=%1d/s" % self.syncRate.value)
        return self.syncRate.value
        
        
        
        
#           
#    def startTTTR(self):
#        
#        print(datetime.now(), ' [HydraHarp 400] TCSPC measurement started')
#        
#
#        
#        self.counts = []
#        progress = 0
#       
#        self.lib.HH_StartMeas(ct.c_int(DEV_NUM), ct.c_int(self.tacq))
#        meas = True
#        self.measure_state = 'measuring'
#        self.ctcstatus = ct.c_int(0)
#        while meas is True:
#            self.lib.HH_GetFlags(ct.c_int(DEV_NUM), byref(self.flags))
#            
#            if self.flags.value & FLAG_FIFOFULL > 0:
#                print("\nFiFo Overrun!")
#                self.stopTTTR()
#            
#            
#            self.lib.HH_ReadFiFo(ct.c_int(DEV_NUM), byref(self.buffer), 
#                                 TTREADMAX, byref(self.nRecords))
#                
#        
#            if self.nRecords.value > 0:
#                #print('[HydraHarp 400]', self.nRecords.value)
#                # We could just iterate through our buffer with a for loop, however,
#                # this is slow and might cause a FIFO overrun. So instead, we shrinken
#                # the buffer to its appropriate length with array slicing, which gives
#                # us a python list. This list then needs to be converted back into
#                # a ctype array which can be written at once to the output file
#                self.counts.append(self.buffer[0:self.nRecords.value])
#                #outputfile.write((ct.c_uint*self.nRecords.value)(*self.buffer[0:self.nRecords.value]))
#                progress += self.nRecords.value
#                #sys.stdout.write("\rProgress:%9u" % progress)
#                #sys.stdout.flush()
##                sys.stdout.write("\rProgress:%9u" % progress)
##                sys.stdout.flush()
#                
#            else:
#                self.lib.HH_CTCStatus(ct.c_int(DEV_NUM), byref(self.ctcstatus))
#                
#                if self.ctcstatus.value > 0: 
#                    print("\nDone")
#                    self.numRecords = progress
#                    self.stopTTTR()
#                    #print('{} events recorded'.format(self.numRecords))
#                    meas = False
#                    self.measure_state = 'done'
#    
#    
    
    
    def startTTTR(self):
        
        #print(datetime.now(), ' [HydraHarp 400] TCSPC measurement started')
        #print("tacq ", self.tacq)
        self.counts = []
        self.ctcdone = False
        #outputfile = open(outputfilename, "wb+") 
        progress = 0
        meas = True
        self.lib.HH_StartMeas(ct.c_int(DEV_NUM), ct.c_int(self.tacq))

        self.measure_state = 'measuring'
        
        
        self.ctcstatus = ct.c_int(0)
        while self.ctcdone is False:
            self.lib.HH_GetFlags(ct.c_int(DEV_NUM), byref(self.flags))
            
            if self.flags.value & FLAG_FIFOFULL > 0:
                print("\nFiFo Overrun!")
                self.stopTTTR()
            
            
            self.lib.HH_ReadFiFo(ct.c_int(DEV_NUM), byref(self.buffer), 
                                 TTREADMAX, byref(self.nRecords))
                
        
            if self.nRecords.value > 0:
                #print('[HydraHarp 400]', self.nRecords.value)
                # We could just iterate through our buffer with a for loop, however,
                # this is slow and might cause a FIFO overrun. So instead, we shrinken
                # the buffer to its appropriate length with array slicing, which gives
                # us a python list. This list then needs to be converted back into
                # a ctype array which can be written at once to the output file
                #outputfile.write((ct.c_uint*self.nRecords.value)(*self.buffer[0:self.nRecords.value]))
                self.counts.append(self.buffer[0:self.nRecords.value])
                progress += self.nRecords.value
                #sys.stdout.write("\rProgress:%9u" % progress)
                #sys.stdout.flush()
                
            else:
                self.lib.HH_CTCStatus(ct.c_int(DEV_NUM), byref(self.ctcstatus))
                
                if self.ctcstatus.value > 0: 
                    #print("\nDone")
                    self.numRecords = progress
                    self.stopTTTRscan()
                    #print('{} events recorded'.format(self.numRecords))
                    self.ctcdone = True


    def startTTTR_mp(self):
        print("mping")
        print("I'm running on CPU #%s" % mp.current_process().name)
        #print(datetime.now(), ' [HydraHarp 400] TCSPC measurement started')
        #print("tacq ", self.tacq)
        self.counts = []
        self.ctcdone = False
        #outputfile = open(outputfilename, "wb+") 
        progress = 0
        meas = True
        self.lib.HH_StartMeas(ct.c_int(DEV_NUM), ct.c_int(self.tacq))

        self.measure_state = 'measuring'
        
        
        self.ctcstatus = ct.c_int(0)
        while self.ctcdone is False:
            self.lib.HH_GetFlags(ct.c_int(DEV_NUM), byref(self.flags))
            
            if self.flags.value & FLAG_FIFOFULL > 0:
                print("\nFiFo Overrun!")
                self.stopTTTR()
            
            
            self.lib.HH_ReadFiFo(ct.c_int(DEV_NUM), byref(self.buffer), 
                                 TTREADMAX, byref(self.nRecords))
                
        
            if self.nRecords.value > 0:
                #print('[HydraHarp 400]', self.nRecords.value)
                # We could just iterate through our buffer with a for loop, however,
                # this is slow and might cause a FIFO overrun. So instead, we shrinken
                # the buffer to its appropriate length with array slicing, which gives
                # us a python list. This list then needs to be converted back into
                # a ctype array which can be written at once to the output file
                #outputfile.write((ct.c_uint*self.nRecords.value)(*self.buffer[0:self.nRecords.value]))
                self.counts.append(self.buffer[0:self.nRecords.value])
                progress += self.nRecords.value
                #sys.stdout.write("\rProgress:%9u" % progress)
                #sys.stdout.flush()
                
            else:
                self.lib.HH_CTCStatus(ct.c_int(DEV_NUM), byref(self.ctcstatus))
                
                if self.ctcstatus.value > 0: 
                    #print("\nDone")
                    self.numRecords = progress
                    self.stopTTTRscan()
                    #print('{} events recorded'.format(self.numRecords))
                    self.ctcdone = True





    def initHistoScan(self):
        MODE_HIST = 0

        self.lib.HH_Initialize(ct.c_int(DEV_NUM), ct.c_int(MODE_HIST), ct.c_int(0))
        
        
        
    def startHistoScan(self):
        self.lib.HH_ClearHistMem(ct.c_int(DEV_NUM))
        self.histcounts = [(ct.c_uint * MAXHISTLEN)() for i in range(0, HHMAXINPCHAN)]
        #print("Hydraharp starts Histoscan")
        self.lib.HH_StartMeas(ct.c_int(DEV_NUM), ct.c_int(self.tacq))    
        self.ctcstatus = ct.c_int(0)
        while self.ctcstatus.value == 0:
            self.lib.HH_CTCStatus(ct.c_int(DEV_NUM), byref(self.ctcstatus))
        
        self.lib.HH_StopMeas(ct.c_int(DEV_NUM))
        
        #for i in range(0, self.numChannels.value):
        i = 0
        
        self.lib.HH_GetHistogram(ct.c_int(DEV_NUM), byref(self.histcounts[i]),\
                                  ct.c_int(i), ct.c_int(0))
        self.integralCount = 0
        for j in range(0, self.histLen.value):
            self.integralCount += self.histcounts[i][j]
        #print("Hydraharp ends Histoscan")
        self.ctcdone = True
        
    def closehh(self):
        self.lib.HH_CloseDevice(ct.c_int(DEV_NUM))
    def stopTTTRscan(self):
        
        self.lib.HH_StopMeas(ct.c_int(DEV_NUM))

    
    
    
    def stopTTTR(self):
        
        self.lib.HH_StopMeas(ct.c_int(DEV_NUM))
        self.lib.HH_CloseDevice(ct.c_int(DEV_NUM))
       
    def initialize(self):
        
        self.lib.HH_Initialize(ct.c_int(DEV_NUM), ct.c_int(self.mode), ct.c_int(0))
        
    def finalize(self):
        print("init HH")
        self.lib.HH_CloseDevice(ct.c_int(DEV_NUM))