# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:25:20 2018

@author: USUARIO
"""

import numpy as np
import configparser
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

def convert(x, key):
    
    # ADC/DAC to Voltage parameters

    m_VtoU = (2**15)/10  #  in V^-1
    q_VtoU = 2**15   

    # piezo voltage-position calibration parameters
    
    m_VtoL = 2.91  # in µm/V
    q_VtoL = -0.02  # in µm

#    m_VtoL = 2 # in µm/V
#    q_VtoL = 0 # in µm
    
    if np.any(x) < 0:
        
        return print('Error: x cannot take negative values')
        
    else:
        
        if key == 'VtoU':
            
            value = x * m_VtoU + q_VtoU
            value = np.around(value, 0)
            
        if key == 'UtoV':
            
            value = (x - q_VtoU)/m_VtoU
            
        if key == 'XtoU':
            
            value = ((x - q_VtoL)/m_VtoL) * m_VtoU + q_VtoU
            value = np.around(value, 0)
            
        if key == 'UtoX':
        
            value = ((x - q_VtoU)/m_VtoU) * m_VtoL + q_VtoL
            
        if key == 'ΔXtoU':
            
            value = (x/m_VtoL) * m_VtoU 
            value = np.around(value, 0)
            
        if key == 'ΔUtoX':
            
            value = (x/m_VtoU) * m_VtoL
            
        if key == 'ΔVtoX':
        
            value = x * m_VtoL
            
        if key == 'VtoX':
            
            value = x * m_VtoL + q_VtoL

        return value
        
        
def ROIscanRelativePOS(pos, Nimage, nROI):
    print("here")
    print(nROI)
    print(Nimage)
    print(pos[1])
    scanPos = np.zeros(2)
    scanPos[0] = pos[0]
    #scanPos[1] = - pos[1] + Nimage - nROI
    scanPos[1] =  pos[1]
    return scanPos
    
    
def timeToADwin(t):
    
    "time in µs to ADwin time units of 3.33 ns"
    
    time_unit = 3.33 * 10**-3  # 3.33 ns
    
    units = np.array(t/(time_unit), dtype='int')
    
    return units
    
def velToADwin(v):
    
    v_adwin = v * (convert(1000, 'ΔXtoU')/timeToADwin(1))
    
    return v_adwin
    
def accToADwin(a):
    
    a_adwin = a * (convert(1000, 'ΔXtoU')/timeToADwin(1)**2)
    
    return a_adwin
    
def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt
    
def saveMINFLUXconfig(main, dateandtime, name, filename=None):

    if filename is None:
        filename = os.path.join(os.getcwd(), name)

    config = configparser.ConfigParser()

    config['Scanning parameters'] = {

        'Date and time': dateandtime,
        'Initial Position [x0, y0, z0] (µm)': main.initialPos,
        'measType': main.measType,
        'Acqtime (s)': main.acqtime}

    with open(filename + '.txt', 'w') as configfile:
        config.write(configfile)

def saveODconfig(main, dateandtime, name):
    
#    print(dateandtime, " [ODconfig] wrote OD config with parameters: ",main.angle, main.OD, main.ODslope, main.ODangleoffset) 
    config = configparser.ConfigParser()
    config['Intensityparameters'] = {
            
            'Date and time': dateandtime,
            'angle': main.angle,
            'OD': main.OD,
            'OD slope': main.ODslope,
            'OD angle offset': main.ODangleoffset
          
                   
          
          }
    print(dateandtime, " [ODconfig] wrote OD config with parameters: ",main.angle, main.OD, main.ODslope, main.ODangleoffset) 
    with open(name + '.txt', 'w') as configfile:
        config.write(configfile)


def readODconfig(main, name):
    config = configparser.ConfigParser()
    
    
    config.read(name + '.txt')
#    print(config)
    dateandtime = config['Intensityparameters']['date and time']

    main.angleinit = float(config['Intensityparameters']['angle'])
    main.ODinit = float(config['Intensityparameters']['OD'])
    main.ODslopeinit = float(config['Intensityparameters']['od slope'])
    main.ODangleoffsetinit = float(config['Intensityparameters']['od angle offset'])
    print(dateandtime, "[ODconfig] Read OD config with parameters: ",main.angleinit, main.ODinit, main.ODslopeinit, main.ODangleoffsetinit) 



def saveConfig(main, dateandtime, name, filename=None):

    if filename is None:
        filename = os.path.join(os.getcwd(), name)

    config = configparser.ConfigParser()

    config['Scanning parameters'] = {

        'Date and time': dateandtime,
        'Initial Position [x0, y0, z0] (µm)': main.initialPos,
        'Scan range (µm)': main.scanRange,
        'Pixel time (µs)': main.pxTime,
        'Number of pixels': main.NofPixels,
        #'a_max (µm/µs^2)': str(main.a_max),
        'Pixel size (µm)': main.pxSize,
        'Frame time (s)': main.frameTime,
        'Scan type': main.scantype}

    with open(filename + '.txt', 'w') as configfile:
        config.write(configfile)




def getUniqueName(name):
    
    n = 1
    while os.path.exists(name + '.txt'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name



def getUniqueFolder(name):
    
    n = 1
    while os.path.exists(name):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name
    
def ScanSignal(scan_range, n_pixels, n_aux_pixels, px_time, a_aux, dy, x_i,
               y_i, z_i, scantype, waitingtime=0):
    
    # derived parameters
    
    n_wt_pixels = int(waitingtime/px_time)
    px_size = scan_range/n_pixels
    v = px_size/px_time
    line_time = n_pixels * px_time
    
    aux_time =  [1, 1, 1,1 ]
    aux_range = [1, 1, 1,1 ]
    
    if np.all(a_aux == np.flipud(a_aux)) or np.all(a_aux[0:2] == a_aux[2:4]):
        
        pass
        
    else:
        
        print(datetime.now(), '[scan-tools] Scan signal has unmatching aux accelerations')
    
    # scan signal 
    
    size = 4 * n_aux_pixels + 2 * n_pixels
    total_range = aux_range[0] + aux_range[1] + scan_range
    
    if total_range > 20:
        print(datetime.now(), '[scan-tools] Warning: scan + aux scan excede DAC/piezo range! ' 
              'Scan signal will be saturated')
    else:
        print(datetime.now(), '[scan-tools] Scan signal OK')
        
    signal_time = np.zeros(size)
    signal_x = np.zeros(size)
    signal_y = np.zeros(size)
    
    # smooth dy part    
    
    signal_y[0:n_aux_pixels] = np.linspace(0, dy, n_aux_pixels)
    signal_y[n_aux_pixels:size] = dy * np.ones(size - n_aux_pixels)    
    
    # part 1
    
    i0 = 0
    i1 = n_aux_pixels
    
    signal_time[i0:i1] = np.linspace(0, aux_time[0], n_aux_pixels)
    
    t1 = signal_time[i0:i1]
    
    signal_x[i0:i1] = (1/2) * a_aux[0] * t1**2
    
    
    # part 2
    
    i2 = n_aux_pixels + n_pixels
    
    signal_time[i1:i2] = np.linspace(aux_time[0], aux_time[0] + line_time, n_pixels)
    
    t2 = signal_time[i1:i2] - aux_time[0]
    x02 = aux_range[0]

    signal_x[i1:i2] = x02 + v * t2
    
    # part 3
    
    i3 = 2 * n_aux_pixels + n_pixels
    
    t3_i = aux_time[0] + line_time
    t3_f = aux_time[0] + aux_time[1] + line_time   
    signal_time[i2:i3] = np.linspace(t3_i, t3_f, n_aux_pixels)
    
    t3 = signal_time[i2:i3] - (aux_time[0] + line_time)
    x03 = aux_range[0] + scan_range
    
    signal_x[i2:i3] = - (1/2) * a_aux[1] * t3**2 + v * t3 + x03
    
    # part 4
    
    i4 = 3 * n_aux_pixels + n_pixels
    
    t4_i = aux_time[0] + aux_time[1] + line_time
    t4_f = aux_time[0] + aux_time[1] + aux_time[2] + line_time   
    
    signal_time[i3:i4] = np.linspace(t4_i, t4_f, n_aux_pixels)
    
    t4 = signal_time[i3:i4] - t4_i
    x04 = aux_range[0] + aux_range[1] + scan_range
    
    signal_x[i3:i4] = - (1/2) * a_aux[2] * t4**2 + x04
    
    # part 5
    
    i5 = 3 * n_aux_pixels + 2 * n_pixels
    
    t5_i = aux_time[0] + aux_time[1] + aux_time[2] + line_time  
    t5_f = aux_time[0] + aux_time[1] + aux_time[2] + 2 * line_time  
    
    signal_time[i4:i5] = np.linspace(t5_i, t5_f, n_pixels)
    
    t5 = signal_time[i4:i5] - t5_i
    x05 = aux_range[3] + scan_range
    
    signal_x[i4:i5] = x05 - v * t5    

    # part 6

    i6 = size

    t6_i = aux_time[0] + aux_time[1] + aux_time[2] + 2 * line_time
    t6_f = np.sum(aux_time) + 2 * line_time

    signal_time[i5:i6] = np.linspace(t6_i, t6_f, n_aux_pixels)

    t6 = signal_time[i5:i6] - t6_i
    x06 = aux_range[3]

    signal_x[i5:i6] = (1/2) * a_aux[3] * t6**2 - v * t6 + x06

    if waitingtime != 0:

        signal_x = list(signal_x)
        signal_x[i3:i3] = x04 * np.ones(n_wt_pixels)

        signal_time[i3:i6] = signal_time[i3:i6] + waitingtime
        signal_time = list(signal_time)
        signal_time[i3:i3] = np.linspace(t3_f, t3_f + waitingtime, n_wt_pixels)
        
        signal_y = np.append(signal_y, np.ones(n_wt_pixels) * signal_y[i3])
        
        signal_x = np.array(signal_x)
        signal_time = np.array(signal_time)
        
    else:
        
        pass
    
    
    
    if scantype == 'xy':
        
        signal_f = signal_x + x_i
        signal_s = signal_y + y_i
    
    if scantype == 'xz':
    
        signal_f = signal_x + x_i
        signal_s = signal_y + (z_i - scan_range/2)
        
    if scantype == 'yz':
        
        signal_f = signal_x + y_i
        signal_s = signal_y + (z_i - scan_range/2)

    return signal_time, signal_f, signal_s

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Plot of covariance ellipse
    
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.
    Returns
    -------
    width(w), height(h), rotation(theta in degrees):
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)
    
    val, vec =  np.linalg.eig(cov)
    order = val.argsort()[::]
    val = val[order]
    vec = vec[order]
    w, h = 2 * np.sqrt(val[:, None] * r2)
    theta = np.degrees(np.arctan2(*vec[::, 0]))
    return w, h, theta
    