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
from scipy import optimize as opt


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
        'Acqtime (s)': main.acqtime
        } #todo OD

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
            name = name + '_1'#insertSuffix_new(name, '_{}'.format(n))
        n += 1

    return name


def getUniqueNameptu(name, date):
    
    n = 1
    while os.path.exists(name + '_' + date + '_arrays.ptu'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = name + '_1' #insertSuffix(name, '_{}'.format(n))
        n += 1

    return name + '_' + date

def getUniqueNameSPAD(name, date):
    
    n = 1
    while os.path.exists(name + '_' + date + '_arrays_spad.ptu'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = name + '_1' #insertSuffix(name, '_{}'.format(n))
        n += 1

    return name + '_' + date



def getUniqueNameSPADAPD(name, date):
    
    n = 1
    while os.path.exists(name + '_' + date + '_arrays_spad.ptu') or os.path.exists(name + '_' + date + '_arrays.ptu'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = name + '_1' #insertSuffix(name, '_{}'.format(n))
        n += 1

    return name + '_' + date



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
    





def gaussianfit(image, spots, size, pxSize):
    """
    Performs a gaussian fit on the image

    Parameters
    ----------
    image - np.array
        
    spot - x,y middle pos
    
    size - +- size is ROI
    
    pxSize - pixel size
    
    
    Return
    ----------
    massCenter - np.array - in pixel
        

    """
    currentx = []
    currenty = []
    print(spots)
#    print(size)
    for spot in spots:
#        print(spot)
        
        
        
        xmin = int(round(spot[0] - size))
        xmax = int(round(spot[0] + size))
        ymin = int(round(spot[1] - size))
        ymax = int(round(spot[1] + size))
        
#        xmin, xmax, ymin, ymax = roicoordinate
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = np.array([xmin, xmax, ymin, ymax]) * pxSize
#        
        # select the data of the image corresponding to the ROI

        array = image[xmin:xmax, ymin:ymax] *10
        
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
    
        xrange = size # in px
        yrange = size # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
#        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')

        x_sub_nm = np.arange(0, xsubsize) * pxSize
        y_sub_nm = np.arange(0, ysubsize) * pxSize

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
#        print(Mx_sub)
        
#        print(array)
        # make initial guess for parameters
        
        # bkg = np.min(array)
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 3 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [[A, x0, y0, σ, σ, bkg]]
        
#        fig, ax = plt.subplots(1, 1) 
#    
#        ax.imshow(array)
        
#        print(initial_guess_G)
        try:
            poptG, pcovG = opt.curve_fit(gaussian2D, (Mx_sub, My_sub), 
                         array.ravel(), p0=initial_guess_G)
            # print(poptG)
            # poptG, pcovG = opt.curve_fit(gaussian2D_jonas, (Mx_sub, My_sub), 
            #                          array.ravel(), p0=initial_guess_G)
            # poptG = np.around(poptG, 3)
    
            [A, x0, y0, σ_x, σ_y, bkg] = poptG
#            print(x0, y0)
            
            
#            print(x0,y0)
#            print( Mx_nm[xmin_id, ymin_id],  My_nm[xmin_id, ymin_id])
#            print(poptG)
            currentx.append(xmin + x0 )#+ Mx_nm[xmin_id, ymin_id])
            currenty.append(ymin + y0 )#+ My_nm[xmin_id, ymin_id])    
        except(RuntimeError, ValueError):
            
            print('[xyz_tracking] Gaussian fit did not work')
            currentx.append(0)
            currenty.append(0)  


        
        # retrieve results

        

    return np.array(currentx), np.array(currenty)




def gaussianfitsigma(image, spots, size, pxSize):
    """
    Performs a gaussian fit on the image

    Parameters
    ----------
    image - np.array
        
    spot - x,y middle pos
    
    size - +- size is ROI
    
    pxSize - pixel size
    
    
    Return
    ----------
    massCenter - np.array - in pixel
    
    sigmas
        

    """
    currentx = []
    currenty = []
    sigmax = []
    sigmay = []

    for spot in spots:
#        print(spot)
        xmin = spot[0] - size
        xmax = spot[0] + size
        ymin = spot[1] - size
        ymax = spot[1] + size
        
#        xmin, xmax, ymin, ymax = roicoordinate
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = np.array([xmin, xmax, ymin, ymax]) * pxSize
#        
        # select the data of the image corresponding to the ROI

        array = image[xmin:xmax, ymin:ymax] *10
        
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
    
        xrange = size # in px
        yrange = size # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
#        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')

        x_sub_nm = np.arange(0, xsubsize) * pxSize
        y_sub_nm = np.arange(0, ysubsize) * pxSize

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
#        print(Mx_sub)
        
#        print(array)
        # make initial guess for parameters
        
        # bkg = np.min(array)
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 3 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [[A, x0, y0, σ, σ, bkg]]
        
#        fig, ax = plt.subplots(1, 1) 
#    
#        ax.imshow(array)
        
#        print(initial_guess_G)
        try:
            poptG, pcovG = opt.curve_fit(gaussian2D, (Mx_sub, My_sub), 
                         array.ravel(), p0=initial_guess_G)
            # print(poptG)
            # poptG, pcovG = opt.curve_fit(gaussian2D_jonas, (Mx_sub, My_sub), 
            #                          array.ravel(), p0=initial_guess_G)
            # poptG = np.around(poptG, 3)
    
            [A, x0, y0, σ_x, σ_y, bkg] = poptG
#            print(x0, y0)
            
            
#            print(x0,y0)
#            print( Mx_nm[xmin_id, ymin_id],  My_nm[xmin_id, ymin_id])
#            print(poptG)
            currentx.append(xmin + x0 )#+ Mx_nm[xmin_id, ymin_id])
            currenty.append(ymin + y0 )#+ My_nm[xmin_id, ymin_id])
            sigmax.append(σ_x)
            sigmay.append(σ_y)
            
        except(RuntimeError, ValueError):
            
            print('[xyz_tracking] Gaussian fit did not work')
            currentx.append(0)
            currenty.append(0)  
            sigmax.append(1000)
            sigmay.append(1000)


        
        # retrieve results

        

    return np.array(currentx), np.array(currenty), np.array(sigmax), np.array(sigmay)


    
def subtract_background( image, radius=15, light_bg=False):
    #From https://forum.image.sc/t/background-subtraction-in-scikit-image/39118/4
    from skimage.morphology import white_tophat, black_tophat, disk
    
    str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)
    
    
    
    
    

def gaussianfit_new(image, spots, size, pxSize):
    """
    Performs a gaussian fit on the image

    Parameters
    ----------
    image - np.array
        
    spot - x,y middle pos
    
    size - +- size is ROI
    
    pxSize - pixel size
    
    
    Return
    ----------
    massCenter - np.array - in pixel
        

    """
    currentx = []
    currenty = []

    for spot in spots:
        print(spot)
        xmin = spot[0] - size
        xmax = spot[0] + size
        ymin = spot[1] - size
        ymax = spot[1] + size
        
#        xmin, xmax, ymin, ymax = roicoordinate
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = np.array([xmin, xmax, ymin, ymax]) * pxSize
#        
        # select the data of the image corresponding to the ROI

        array = image[xmin:xmax, ymin:ymax] *10
        
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
    
        xrange = size # in px
        yrange = size # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
#        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')

        x_sub_nm = np.arange(0, xsubsize) * pxSize
        y_sub_nm = np.arange(0, ysubsize) * pxSize

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
#        print(Mx_sub)
        
#        print(array)
        # make initial guess for parameters
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 100 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
        
#        fig, ax = plt.subplots(1, 1) 
#    
#        ax.imshow(array)
        
#        print(initial_guess_G)
        try:
            poptG, pcovG = opt.curve_fit(gaussian2D, (Mx_sub, My_sub), 
                                     array.ravel(), p0=initial_guess_G)
            poptG = np.around(poptG, 2)
    
            A, x0, y0, σ_x, σ_y, bkg = poptG
            currentx.append(x0 + Mx_nm[xmin_id, ymin_id])
            currenty.append(y0 + My_nm[xmin_id, ymin_id])    
        except(RuntimeError, ValueError):
            
            print('[xyz_tracking] Gaussian fit did not work')
            currentx.append(0)
            currenty.append(0)  


        
        # retrieve results

        

    return np.array(currentx), np.array(currenty)

def calculate_waveletAuNR( input,max_level,k):
    '''calculate a trous wavelet based on olivo martin 2002'''
    from scipy.ndimage import convolve1d
    from skimage.filters import gaussian,threshold_otsu
    input = gaussian(input)
    import numpy as np
    data_size = len(input)
    out = np.zeros((1080,1440,max_level)) #TODO dont hardcode
    out[:,:,0] = input
    for i in range(1,max_level):
        kernel = np.concatenate(([1/16],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[3/8],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[1/16]))
        wavelet = convolve1d(input,kernel,mode="reflect",axis=0)
        wavelet = convolve1d(wavelet,kernel,mode="reflect",axis=1)
        wavelet = wavelet-out[:,:,i-1]
        abs = np.abs(wavelet)
        wavelet[abs<k*np.median(wavelet)] = 0
        out[:,:,i] = wavelet
    final_wavelet = np.abs(np.prod(out[:,:,1:],axis=2))
    final_wavelet = np.ma.log10(final_wavelet).filled(0)
    final_wavelet[final_wavelet<threshold_otsu(final_wavelet[final_wavelet!=0])]=0
    return final_wavelet



        
def calculate_wavelet( input,max_level,k):
    '''calculate a trous wavelet based on olivo martin 2002'''
    from scipy.ndimage import convolve1d
    from skimage.filters import gaussian,threshold_otsu
    input = gaussian(input)
    import numpy as np
    data_size = len(input)
    out = np.zeros((data_size,data_size,max_level))
    out[:,:,0] = input
    for i in range(1,max_level):
        kernel = np.concatenate(([1/16],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[3/8],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[1/16]))
        wavelet = convolve1d(input,kernel,mode="reflect",axis=0)
        wavelet = convolve1d(wavelet,kernel,mode="reflect",axis=1)
        wavelet = wavelet-out[:,:,i-1]
        abs = np.abs(wavelet)
        wavelet[abs<k*np.median(wavelet)] = 0
        out[:,:,i] = wavelet
    final_wavelet = np.abs(np.prod(out[:,:,1:],axis=2))
    final_wavelet = np.ma.log10(final_wavelet).filled(0)
    final_wavelet[final_wavelet<threshold_otsu(final_wavelet[final_wavelet!=0])]=0
    return final_wavelet

def gaussian2D(grid, amplitude, x0, y0, σ_x, σ_y, offset, theta=0):
    
    # TO DO (optional): change parametrization to this one
    # http://mathworld.wolfram.com/BivariateNormalDistribution.html  
    # supposed to be more robust for the numerical fit
    
    x, y = grid
    x0 = float(x0)
    y0 = float(y0)   
    a = (np.cos(theta)**2)/(2*σ_x**2) + (np.sin(theta)**2)/(2*σ_y**2)
    b = -(np.sin(2*theta))/(4*σ_x**2) + (np.sin(2*theta))/(4*σ_y**2)
    c = (np.sin(theta)**2)/(2*σ_x**2) + (np.cos(theta)**2)/(2*σ_y**2)
    G = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return G.ravel()
    