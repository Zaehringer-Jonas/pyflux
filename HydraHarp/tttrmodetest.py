# Demo for access to HydraHarp 400 Hardware via HHLIB.DLL v 3.0.
# The program performs a measurement based on hard coded settings.
# The resulting data is stored in a binary output file.
#
# Keno Goertz, PicoQuant GmbH, February 2018

import time
import ctypes as ct
from ctypes import byref
import sys
import struct
import os
import numpy as np
import io


if sys.version_info[0] < 3:
    print("[Warning] Python 2 is not fully supported. It might work, but "
          "use Python 3 if you encounter errors.\n")
    raw_input("press RETURN to continue"); print
    input = raw_input

# From hhdefin.h
LIB_VERSION = "3.0"
MAXDEVNUM = 4
MODE_T2 = 2
MODE_T3 = 3
MAXLENCODE = 6
HHMAXINPCHAN = 4
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0001
FLAG_FIFOFULL = 0x0002

# Measurement parameters, these are hardcoded since this is just a demo
mode = MODE_T3 # set T2 or T3 here, observe suitable Syncdivider and Range!
binning = 3 # you can change this, meaningful only in T3 mode
offset = 1 # you can change this, meaningful only in T3 mode
tacq = 2000  # Measurement time in millisec, you can change this
syncDivider = 2 # you can change this, observe mode! READ MANUAL!
syncCFDZeroCross = 10 # you can change this (in mV)
syncCFDLevel = 50 # you can change this (in mV)
syncChannelOffset = -5000 # you can change this (in ps, like a cable delay)
inputCFDZeroCross = 10 # you can change this (in mV)
inputCFDLevel = 50 # you can change this (in mV)
inputChannelOffset = 0 # you can change this (in ps, like a cable delay)

# Variables to store information read from DLLs
buffer = (ct.c_uint * TTREADMAX)()
dev = []
libVersion = ct.create_string_buffer(b"", 8)
hwSerial = ct.create_string_buffer(b"", 8)
hwPartno = ct.create_string_buffer(b"", 8)
hwVersion = ct.create_string_buffer(b"", 8)
hwModel = ct.create_string_buffer(b"", 16)
errorString = ct.create_string_buffer(b"", 40)
numChannels = ct.c_int()
resolution = ct.c_double()
syncRate = ct.c_int()
countRate = ct.c_int()
flags = ct.c_int()
nRecords = ct.c_int()
ctcstatus = ct.c_int()
warnings = ct.c_int()
warningstext = ct.create_string_buffer(b"", 16384)

if os.name == "nt":
    hhlib = ct.WinDLL("hhlib64.dll")
else:
    hhlib = ct.CDLL("libhh400.so")

def closeDevices():
    for i in range(0, MAXDEVNUM):
        hhlib.HH_CloseDevice(ct.c_int(i))
    exit(0)

def stoptttr():
    retcode = hhlib.HH_StopMeas(ct.c_int(dev[0]))
    if retcode < 0:
        print("HH_StopMeas error %1d. Aborted." % retcode)
    print("beforeclose")
    closeDevices()

def tryfunc(retcode, funcName, measRunning=False):
    if retcode < 0:
        hhlib.HH_GetErrorString(errorString, ct.c_int(retcode))
        print("HH_%s error %d (%s). Aborted." % (funcName, retcode,\
              errorString.value.decode("utf-8")))
        if measRunning:
            stoptttr()
        else:
            closeDevices()

hhlib.HH_GetLibraryVersion(libVersion)
print("Library version is %s" % libVersion.value.decode("utf-8"))
if libVersion.value.decode("utf-8") != LIB_VERSION:
    print("Warning: The application was built for version %s" % LIB_VERSION)

outputfile = open("testfile.out", "wb+", ) 

print("\nMode             : %d" % mode)
print("Binning           : %d" % binning)
print("Offset            : %d" % offset)
print("AcquisitionTime   : %d" % tacq)
print("SyncDivider       : %d" % syncDivider)
print("SyncCFDZeroCross  : %d" % syncCFDZeroCross)
print("SyncCFDLevel      : %d" % syncCFDLevel)
print("InputCFDZeroCross : %d" % inputCFDZeroCross)
print("InputCFDLevel     : %d" % inputCFDLevel)

print("\nSearching for HydraHarp devices...")
print("Devidx     Status")

for i in range(0, MAXDEVNUM):
    retcode = hhlib.HH_OpenDevice(ct.c_int(i), hwSerial)
    if retcode == 0:
        print("  %1d        S/N %s" % (i, hwSerial.value.decode("utf-8")))
        dev.append(i)
    else:
        if retcode == -1: # HH_ERROR_DEVICE_OPEN_FAIL
            print("  %1d        no device" % i)
        else:
            hhlib.HH_GetErrorString(errorString, ct.c_int(retcode))
            print("  %1d        %s" % (i, errorString.value.decode("utf8")))

# In this demo we will use the first HydraHarp device we find, i.e. dev[0].
# You can also use multiple devices in parallel.
# You can also check for specific serial numbers, so that you always know 
# which physical device you are talking to.

if len(dev) < 1:
    print("No device available.")
    closeDevices()
print(dev)
print("Using device #%1d" % dev[0])
print("\nInitializing the device...")

# with internal clock
tryfunc(hhlib.HH_Initialize(ct.c_int(dev[0]), ct.c_int(mode), ct.c_int(0)),\
        "Initialize")

# Only for information
tryfunc(hhlib.HH_GetHardwareInfo(dev[0], hwModel, hwPartno, hwVersion),\
        "GetHardwareInfo")
print("Found Model %s Part no %s Version %s" % (hwModel.value.decode("utf-8"),\
      hwPartno.value.decode("utf-8"), hwVersion.value.decode("utf-8")))

tryfunc(hhlib.HH_GetNumOfInputChannels(ct.c_int(dev[0]), byref(numChannels)),\
        "GetNumOfInputChannels")
print("Device has %i input channels." % numChannels.value)

print("\nCalibrating...")
tryfunc(hhlib.HH_Calibrate(ct.c_int(dev[0])), "Calibrate")
tryfunc(hhlib.HH_SetSyncDiv(ct.c_int(dev[0]), ct.c_int(syncDivider)), "SetSyncDiv")

tryfunc(
    hhlib.HH_SetSyncCFD(ct.c_int(dev[0]), ct.c_int(syncCFDLevel),
                        ct.c_int(syncCFDZeroCross)),\
    "SetSyncCFD"
)

tryfunc(hhlib.HH_SetSyncChannelOffset(ct.c_int(dev[0]), ct.c_int(syncChannelOffset)),\
        "SetSyncChannelOffset")

# we use the same input settings for all channels, you can change this
for i in range(0, numChannels.value):
    tryfunc(
        hhlib.HH_SetInputCFD(ct.c_int(dev[0]), ct.c_int(i), ct.c_int(inputCFDLevel),\
                             ct.c_int(inputCFDZeroCross)),\
        "SetInputCFD"
    )

    tryfunc(
        hhlib.HH_SetInputChannelOffset(ct.c_int(dev[0]), ct.c_int(i),\
                                       ct.c_int(inputChannelOffset)),\
        "SetInputChannelOffset"
    )

# Meaningful only in T3 mode
if mode == MODE_T3:
    tryfunc(hhlib.HH_SetBinning(ct.c_int(dev[0]), ct.c_int(binning)), "SetBinning")
    tryfunc(hhlib.HH_SetOffset(ct.c_int(dev[0]), ct.c_int(offset)), "SetOffset")
    tryfunc(hhlib.HH_GetResolution(ct.c_int(dev[0]), byref(resolution)), "GetResolution")
    print("Resolution is %1.1lfps" % resolution.value)

# Note: after Init or SetSyncDiv you must allow >100 ms for valid  count rate readings
time.sleep(0.2)

tryfunc(hhlib.HH_GetSyncRate(ct.c_int(dev[0]), byref(syncRate)), "GetSyncRate")
print("\nSyncrate=%1d/s" % syncRate.value)

for i in range(0, numChannels.value):
    tryfunc(hhlib.HH_GetCountRate(ct.c_int(dev[0]), ct.c_int(i), byref(countRate)),\
            "GetCountRate")
    print("Countrate[%1d]=%1d/s" % (i, countRate.value))

progress = 0
sys.stdout.write("\nProgress:%9u" % progress)
sys.stdout.flush()

tryfunc(hhlib.HH_StartMeas(ct.c_int(dev[0]), ct.c_int(tacq)), "StartMeas")
counts = []
flag = True
while flag:
    tryfunc(hhlib.HH_GetFlags(ct.c_int(dev[0]), byref(flags)), "GetFlags")
    
    if flags.value & FLAG_FIFOFULL > 0:
        print("\nFiFo Overrun!")
        stoptttr()
    
    tryfunc(
        hhlib.HH_ReadFiFo(ct.c_int(dev[0]), byref(buffer), TTREADMAX,\
                          byref(nRecords)),\
        "ReadFiFo", measRunning=True
    )

    if nRecords.value > 0: #Damit nicht Fehler kommen...
        #print(nRecords.value )
        # We could just iterate through our buffer with a for loop, however,
        # this is slow and might cause a FIFO overrun. So instead, we shrinken
        # the buffer to its appropriate length with array slicing, which gives
        # us a python list. This list then needs to be converted back into
        # a ctype array which can be written at once to the output file
        print("\n")
        print(buffer[nRecords.value])
        counts.append(buffer[0:nRecords.value])
        #outputfile.write((ct.c_uint*nRecords.value)(*buffer[0:nRecords.value]))
        sys.stdout.write("\rnRecords.value:%9u" % nRecords.value)
        progress += nRecords.value
        sys.stdout.write("\rProgress:%9u" % progress)
        sys.stdout.flush()
        
    else:

        tryfunc(hhlib.HH_CTCStatus(ct.c_int(dev[0]), byref(ctcstatus)),\
                "CTCStatus")

        if ctcstatus.value > 0: 
            print("\nDone")
            numRecords = progress
            stoptttr()
            print("t1")
            time.sleep(1)
            flag = False
    # within this loop you can also read the count rates if needed.



outputfile.close()

print("test")
print("nRecordsvalue",nRecords.value)

print(counts)

print("test")

flattened = [val for sublist in counts for val in sublist]
print(flattened)

if True:
    oflcorrection = 0
    dlen = 0
    T3WRAPAROUND = 1024
    ntries = 0
    dtime_array = np.zeros(numRecords)
    truensync_array = np.zeros(numRecords)

    for recNum in range(0, len(flattened)):
        if True:
            tempvalue = bin(flattened[recNum])[2:]
            if len(tempvalue) < 32:
                recordData = (32 - len(tempvalue))*"0"+ tempvalue #mit 0 auffuellen
            else:
                recordData = tempvalue
            ntries += 1

            print("\n")
            print("ntries = ", ntries)
        else:
            print("The file ended earlier than expected, at record %d/%d %d."\
                 % (recNum, numRecords, dlen))
            #return dtime_array, truensync_array #TODO:NAJA

            #exit(0)
        print(recordData)
        print(len(recordData))
        print(type(recordData))
        special = int(recordData[0:1], base=2)
        channel = int(recordData[1:7], base=2)
        dtime = int(recordData[7:22], base=2)
        nsync = int(recordData[22:32], base=2)
        
        
        print(recordData)
        print(len(recordData))
        print(type(recordData))
        
        if special == 1:
            if channel == 0x3F: # Overflow
                # Number of overflows in nsync. If 0 or old version, it's an
                # old style single overflow
                if nsync == 0:
                    oflcorrection += T3WRAPAROUND
                    #print("%u OFL * %2x\n" % (recNum, 1))
                else:
                    oflcorrection += T3WRAPAROUND * nsync
                    #print("%u OFL * %2x\n" % (recNum, 1))
            if channel >= 1 and channel <= 15: # markers
                truensync = oflcorrection + nsync
                dtime_array[recNum] = dtime
                truensync_array[recNum] = truensync
        else: # regular input channel
            truensync = oflcorrection + nsync
            #print("%u CHN %1x %u %8.0lf %10u\n" % (recNum, channel,\
            #truensync, (truensync * 1 * 1e9), dtime)) #TODO changeglobRes
            
            dtime_array[recNum] = dtime
            truensync_array[recNum] = truensync
            
            dlen += 1
            #gotPhoton(truensync, channel, dtime)
        if recNum % 100000 == 0:
            sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(numRecords)))
            
            sys.stdout.flush()


print(dtime_array)
print(truensync_array)
datasize = np.size(truensync_array[truensync_array != 0])
data = np.zeros((2, datasize))
data[0, :] = dtime_array[truensync_array != 0]
data[1, :] = truensync_array[truensync_array != 0]
np.savetxt("testi.txt", data.T)
np.savetxt("testi2.txt", flattened)
time.sleep(20)

import Read_PTU
import numpy as np
inputfile = open("testfile", "rb")
"""relTime, absTime = Read_PTU.readHT3(inputfile, progress)


datasize = np.size(absTime[absTime != 0])
data = np.zeros((2, datasize))
data[0, :] = relTime[absTime != 0]
data[1, :] = absTime[absTime != 0]
np.savetxt("test.txt", data.T)

#print(absTime)
print("test2")
time.sleep(15)
"""