# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:42:11 2020

@author: student-NBS
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from scipy import optimize as opt
import os
from scipy import ndimage as ndi
import scipy.special as ssp
import iminuit as iminuit
import sys
from datetime import date, datetime
from PIL import Image

def initEBP(middle_beam = 0, color = ''):
    name_PSF = 'PSF_' #name of output PSF tifs
    
    
    datetod = str(date.today()).replace('-', '') #
#    datetod = "20210326"#
    folder = r'C:\Users\MINFLUX\Desktop\Images_MINFLUX'

    subfolder_PSF = datetod + '_EBP\\' 
#    print("loading EBP dir", color)
    #PSF inputs
     #index of beam placed centrally in EBP
    FOV_size = 200 #nm, FOV around the middle beam where PSF is fitted
    picture_size = 400 #nm, picture size of experimental PSFs
    PSF_check = True #if true, images of fitted PSFs, EBP, etc are shown
    #    os.listdir(os.path.join(folder, subfolder_PSF))
    directories= [d for d in os.listdir(os.path.join(folder, subfolder_PSF)) if os.path.isdir(os.path.join(folder, subfolder_PSF,d))]
    
    filelist = []
    filelist_tif = []
    if color == '':
        psfpath = os.path.join(folder, subfolder_PSF, directories[-1])    
    else:
        psfpath = os.path.join(folder, subfolder_PSF, directories[-1],color)
    
    
    print("loading EBP dir", psfpath)
    for file in os.listdir(psfpath):
        if file.endswith("PSF.tif"):
            filelist.append(os.path.join(file))
        if file.endswith(".tif"):
            filelist_tif.append(os.path.join(file))
            
    if len(filelist) == 4:
        PSFinit = False
    else: 
        if len(filelist_tif) == 4:
            PSFinit = True
        else: 
            print('Error when initialising PSFs. \nDelete all TIFs but the four PSFs from PSF folder before reinitialising program!')
            raise FileNotFoundError()


    labels = [200, 455, 710, 965]  
    markers = ['yo', 'ro', 'ko', 'bo', 'go']
    aopt_values = np.zeros((4,27))
    psf_values = np.zeros((4, picture_size, picture_size))
    
    if PSFinit == True: #PSF must be initialised
    
        #initial fit to find beam centres in EBP, create first approximation of fit parameters    
        [x0, y0, aopt, psf] = PSF_fit(psfpath, picture_size, 4, FOV_size, 1, 5.0, picture_check = PSF_check)
        
        # create arrays to fit PSF in FOV around middle beam
        # psf_new: exp PSFs cropped in FOV around middle beam
        x0_final = x0 - x0[middle_beam] + FOV_size/2
        y0_final = y0 - y0[middle_beam] + FOV_size/2
        value_start = np.array([x0[middle_beam], y0[middle_beam]]) - FOV_size/2 
        value_end= np.array([x0[middle_beam], y0[middle_beam]]) + FOV_size/2
        psf_new = psf[:,int(value_start[1]):int(value_end[1]), int(value_start[0]):int(value_end[0])]
        
        #refit PSFs around middle beam but keep x0 and y0 fixed to priorly determined values
        [psf_fit, aopt_new] = curve_fit2(psf_new, x0_final, y0_final, aopt, 4)
        print("fit done")
#        if PSF_check == True:  #plot resulting fits to check accuracy
#            fig10 = plt.figure(figsize=(4, 8))
#            for i in np.arange(4):
#                ax10  = fig10.add_subplot(4,2,i*2+1)   
#                ax10.imshow(psf_new[i])
#                ax10  = fig10.add_subplot(4,2,i*2+2)   
#                ax10.imshow(psf_fit[i])
#            plt.show()
#        print("check done")     
        for i in np.arange(4):
            print(i)
            psfimage = Image.fromarray(psf_fit[i])
            psfimage.save(os.path.join(psfpath, name_PSF+str(i)+"_PSF.tif"))   
    
        
        x0_final = x0 - x0[middle_beam] + FOV_size/2
        y0_final = y0 - y0[middle_beam] + FOV_size/2
        print(x0_final,y0_final)
        
        PSF = psf_fit
        
        #save EBP to config file
        np.savetxt(os.path.join(psfpath, name_PSF+"config.txt"),[x0_final,y0_final])
    
    
    else: #PSF already initialised, just has to be loaded
        filelist.sort()
        psfexp = np.zeros((4, FOV_size, FOV_size))  
        j = 0
        for image in filelist:
            im = io.imread(os.path.join(psfpath, image))
            imarray = np.array(im)
            psfexp[j,:,:] = imarray.astype(float)
            j = j +1
        PSF = psfexp
        
        #load EBP
        coords = np.loadtxt(os.path.join(psfpath, name_PSF+"config.txt"), unpack=True)
        x0_final = coords[:,0]
        y0_final = coords[:,1]
        print("EBP load done")
    return PSF, x0_final, y0_final 

def PSF_fit(psf_folder, picture_size, K, size_psf, pxg, divider, coord = None, picture_check = False):
    
    """   
    Open PSF images from PSF measurement and fit with poly_func()
    Input
    ----------
        psf_folder: Folder containing PSF files
        picture_size: size of PSF images [nm]
        K: number of distinct doughnuts, usually K=4   
        size_psf: size of PSFs calculated around minima [nm]
        pxg: Pixel size of fitted function
        divider: Raw PSF images will be cropped 1/divider on each image side
        coord: drift data if available, (3,?) array
        picture_check: Boolean determining whether png of the PSF fits should be shown and saved
    
    Returns
    -------
        PSF : (4, sizeg, sizeg) array, function from fit evaluated over grid
        x0, y0 : arrays, coordinates of EBP centers
        index : array, coordinates of EBP centers in indexes
        aopt : fit parameters, coefficients of polynomial function
       
    """
    
    # open tiff stack with exp PSF images
    filelist = []
    for file in os.listdir(psf_folder):
        if file.endswith(".tif"):
            filelist.append(os.path.join(file))
    filelist.sort()
    print(filelist)    
    im = io.imread(os.path.join(psf_folder, filelist[0]))
    number_pixels = im.shape[1] #number of px in frame
    pixel_size = picture_size/number_pixels #size per pixel in nm
    psfexp = np.zeros((K, number_pixels, number_pixels))  
    j=0

    for image in filelist:
        print(image)
        im = io.imread(os.path.join(psf_folder, image))
        imarray = np.array(im)
        
        
        if imarray.shape[0] == 2:
            psfexp[j,:,:] = imarray.astype(float)[0,:,:] 
            psfexp[j,-1,-1] = 0  
            j = j + 1
        else:
            psfexp[j,:,:] = imarray.astype(float)[:,:] 
            psfexp[j,-1,-1] = 0  
            j = j + 1
        
    # interpolation to have 1 nm px and correct drift if driftdata is available
    psf = np.zeros((K, picture_size, picture_size)) #fitted to 1 nm pixels  
    for i in np.arange(K):
        psf[i] = ndi.interpolation.zoom(psfexp[i,:,:], pixel_size)
        if coord != None:
            deltax = coord[1, i] - coord[1, 0]
            deltay = coord[2, i] - coord[2, 0]
            psf[i] = ndi.interpolation.shift(psf[i], [deltax, deltay])
        
    # crop borders to avoid artifacts 
    border = int(picture_size/divider)
    psfTc = psf[:, border:picture_size-border, border:picture_size-border]  
      
    # spatial grid
    sizeg = np.size(psfTc, axis = 1)
    x = np.arange(0, sizeg, pxg)
    y = np.arange(0, sizeg, pxg)
    x, y = np.meshgrid(x, y) #creates sizeg*sizeg matrix of indexes for x and y
    
    # fit PSFs  with poly_func and find central coordinates (x0, y0)
    PSF = np.zeros((K, sizeg, sizeg))
    x0 = np.zeros(K)
    y0 = np.zeros(K)
    index = np.zeros((K, 2))
    aopt = np.zeros((K, 27))
    
    for i in np.arange(K):
        # initial values for fit parameters x0,y0 and c00
        init = np.unravel_index(np.argmin(psfTc[i, :, :], 
            axis=None), psfTc[i, :, :].shape)
        x0i = init[1]
        y0i = init[0]
        c00i = np.min(psfTc[i, :, :])        
        p0i = [x0i, y0i, c00i, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt[i,:], cov = opt.curve_fit(poly_func, (x,y), psfTc[i, :, :].ravel(), p0 = p0i)    
        q = poly_func((x,y), *aopt[i,:])   
        PSF[i, :, :] = np.reshape(q, (sizeg, sizeg))
        # find min value for each fitted function (EBP centers)
        ind = np.unravel_index(np.argmin(PSF[i, :, :], 
            axis=None), PSF[i, :, :].shape)

        x0[i] = ind[1]
        y0[i] = ind[0]  
        index[i,:] = ind
    
    #saves first approximation zero_values    
    x0_old = x0
    y0_old = y0
    
    # crop the four frames in a FOV of size_psf around the fitted minima
    shift = np.zeros((K, 2))
    value_start = np.zeros((K, 2))
    value_end = np.zeros((K, 2))
    psf_new = np.zeros((K, size_psf, size_psf))      
    for i in np.arange(K):
        shift[i][0] = picture_size/divider + x0_old[i]  # shift in x
        shift[i][1] = picture_size/divider + y0_old[i]  # shift in y
        value_start[i] = shift[i] - size_psf/2
        value_end[i] = shift[i] + size_psf/2
        
        psf_new[i] = psf[i, int(value_start[i][1]):int(value_end[i][1]),
                            int(value_start[i][0]):int(value_end[i][0])]
    
    # refit cropped psf_new with polyfunc and plot pictures to check accuracy
    PSF = np.zeros((K, size_psf, size_psf)) #refitted PSFs
    x0 = np.zeros(K) #minima of refitted PSFs
    y0 = np.zeros(K) #minima of refitted PSFs
    x0_new = np.zeros(K) #minima of refitted PSFs in old FOV
    y0_new = np.zeros(K) #minima of refitted PSFs in old FOV
    index = np.zeros((K, 2))
    aopt = np.zeros((K, 27))
    x = np.arange(0, size_psf, pxg)
    y = np.arange(0, size_psf, pxg)
    x, y = np.meshgrid(x, y) #creates sizeg*sizeg matrix of indexes for x and y
    
    if picture_check == True:
        fig = plt.figure(figsize=(6, 8))
    
    for i in np.arange(K):
        
        # initial values for fit parameters x0,y0 and c00
        init = np.unravel_index(np.argmin(psf_new[i, :, :], 
            axis=None), psf_new[i, :, :].shape)
        x0i = init[1]
        y0i = init[0]
        c00i = np.min(psf_new[i, :, :])        
        p0i = [x0i, y0i, c00i, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt[i,:], cov = opt.curve_fit(poly_func, (x,y), psf_new[i, :, :].ravel(), p0 = p0i) 
        q = poly_func((x,y), *aopt[i,:])   
        PSF[i, :, :] = np.reshape(q, (size_psf, size_psf))
        
        # find min value for each fitted function (EBP centers)
        ind = np.unravel_index(np.argmin(PSF[i, :, :], 
            axis=None), PSF[i, :, :].shape)
        x0[i] = ind[1]
        y0[i] = ind[0] 
        x0_new[i] = x0[i] + value_start[i, 0]
        y0_new[i] = y0[i] + value_start[i, 1]
        index[i,:] = ind
        aopt[i,0] = aopt[i,0] - x0[i]
        aopt[i,1] = aopt[i,1] - y0[i]
       
        #plot fitted functions to check accuracy    
        if picture_check == True:
            
            #grid to show which area is chosen for refit
            grid_1 = [value_end[i][1], value_end[i][1]]
            grid_2 = [value_end[i][0], value_end[i][0]]
            grid_3 = [value_start[i][1], value_start[i][1]]
            grid_4 = [value_start[i][0], value_start[i][0]]
            grid_5 = [value_start[i][0], value_end[i][0]]
            grid_6 = [value_start[i][1], value_end[i][1]]

            # plot whole PSF with grid chosen for refit, 
            # first approximated minimum (red) and final minimum (yellow)
            ax = fig.add_subplot(4,3,i*3+1)
            ax.imshow(psf[i])
            ax.plot(shift[i][0], shift[i][1] , 'ro', markersize = 1)
            ax.plot(grid_5, grid_1, 'b-')
            ax.plot(grid_5, grid_3, 'b-')
            ax.plot(grid_2, grid_6, 'b-')
            ax.plot(grid_4, grid_6, 'b-')
            ax.plot(x0_new[i], y0_new[i], 'yo', markersize = 1)

            #plot area of refitted PSF
            ax = fig.add_subplot(4,3,i*3+2)
            ax.imshow(psf_new[i])
            ax.plot(size_psf/2,size_psf/2, 'ro', markersize = 1)
            ax.plot(x0[i], y0[i], 'yo', markersize = 1)

            #plot fit of refitted PSF
            ax = fig.add_subplot(4,3,i*3+3)
            ax.imshow(PSF[i])
            ax.plot(size_psf/2,size_psf/2, 'ro', markersize = 1)
            ax.plot(x0[i], y0[i], 'yo', markersize = 1)

    if picture_check == True:
        plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
        plt.savefig(psf_folder + 'PSFs.png', format='png', dpi=1000)

    return x0_new, y0_new, aopt, psf 


def poly_func(grid, x0, y0, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
              c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
              c40, c41, c42, c43, c44):
    
    """    
    Polynomial function to fit PSF.
    Uses built-in function polyval2d.
    
    Inputs
    ----------
        grid : x,y array
        x0, y0: centre coordinates
        cij : coefficients of the polynomial function
    Returns
    -------
        q : polynomial evaluated on grid
    
    """

    x, y = grid
    c = np.array([[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14], 
                  [c20, c21, c22, c23, c24], [c30, c31, c32, c33, c34],
                  [c40, c41, c42, c43, c44]])
            
    q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)

    return q.ravel()


def curve_fit2(psf, x0, y0, aopt, K):
    
    """   
    Refits PSFs in desired FOV around middle beam with poly_func and fixed x0,y0
    Input
    ----------
        psf: experimental PSFs cropped in  desired FOV to be analysed
        x0, y0: coordinates of fixed EBP centres
        aopt: parameters of initial fit
              used as initial guess in polynomial function
        K: number of distinct doughnuts, usually K=4
    
    Returns
    -------
        PSF_fit : fitted function in desired FOV
        aopt : fit parameters, coefficients of polynomial function
    
    """
    
    aopt_old = aopt[:,2:] #deletes x0, y0 from aopt
    size_psf = len(psf[0, 0])
    PSF_fit = np.zeros((K, size_psf, size_psf))
    aopt_new = np.zeros((K, aopt.shape[1]))
    x = np.arange(0, size_psf, 1)
    y = np.arange(0, size_psf, 1)
    x, y = np.meshgrid(x, y)
    for i in np.arange(K):
        
        #fit PSFs with priorly determined fixed x0,y0
        aopt_new[i,2:], cov = opt.curve_fit(lambda grid, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
              c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
              c40, c41, c42, c43, c44: poly_func(grid, x0[i], y0[i], c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
              c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
              c40, c41, c42, c43, c44), (x,y), psf[i, :, :].ravel(), p0 = aopt_old[i])
        aopt_new[i,0] = x0[i] #adds x0, y0 back to aopt
        aopt_new[i,1] = y0[i]
        
        #aopt_new[i,2] = 0
        q = poly_func((x,y),*aopt_new[i,:])
        PSF_fit[i, :, :] = np.reshape(q, (size_psf, size_psf)) 
        
    return PSF_fit, aopt_new

def open_TCSPC(folder, subfolder, name):
    
    """   
    Open exp TCSPC data
    Input
    ----------   
        supfolder
        folder   
        subfolder 
        name : file name of experiment
    
    Returns
    -------    
        absTime : array, absTime tags of collected photons
        relTime : array, relTime tags of collected photon 
        channel : array, channel tags of collected photons (=originating APD)
    """
    
    # change dir to dir where TCSPC data is located
    folder =  str(folder)
    subfolder = str(subfolder)  
    newpath = os.path.join(folder, subfolder)   
    os.chdir(newpath)
    
    # open txt file with TCSPC data
    tcspcfile = str(name) + '.txt'
    print(tcspcfile)
    coord = np.loadtxt(tcspcfile, unpack=True)
    channel = coord[2, :]
    absTime = coord[1, :]
    relTime = coord[0, :]
    
    globRes = 1e-3 # absTime resolution 
    timeRes = 1 # relTime resolution (it's already in ns)
    
    absTime = absTime * globRes
    relTime = relTime * timeRes
    
    return absTime, relTime, channel   

def nMINFLUX(τ, relTime, a, b):
    
    """
    Photon collection in a MINFLUX experiment
    (n0, n1, n2, n3)
    
    Inputs
    ----------
    τ : array, times of EBP pulses (1, K)
    relTime : photon arrival times relative to sync (N)
    a : init of temporal window (in ns)
    b : the temporal window lenght (in ns)    
    a,b can be adapted for different lifetimes

    Returns
    -------
    n : (1, K) array acquired photon collection.
    relTimes : array, reltimes of acquired photon collection.
    """
    
    K = 4   #number of beams
    n = np.zeros(K)    # number of photons in each exposition
    relTimes = []    
    for i in np.arange(K):
        ti = τ[i] + a
        tf = τ[i] + a + b
        r = relTime[(relTime>ti) & (relTime<tf)]
        relTimes.append(r)
        n[i] = np.size(r)
    
    return n,np.array(relTimes)


def indexToSpace(index, size, px):

    space = np.zeros(2)
    offset = [size/2, size/2]
    space[1] = - index[0] * px + offset[0]
    space[0] = index[1] * px - offset[1]

    return np.array(space)

def det_SBR(n_phot, n_bckg):
    '''
    determines SBR for different beam positions individually

    Parameters
    ----------
    n_phot : array(4) number of measured photons for four beam positions
    n_bckg : array(4) number of bckg photons for four beam positions

    Returns
    -------
    SBR : array(4), SBR for four beam positions
    '''

    SBR = (n_phot-n_bckg)/n_bckg
    
    return SBR




def pos_MINFLUX(n, PSF, SBR = False, SBR_init = False, ratios = False):
    
    """    
    MINFLUX position estimator (using MLE)
    
    Inputs
    ----------
    n : acquired photon collection (K)
    PSF : array with EBP (K x size x size)
    SBR : estimated (exp) Signal to Bkgd Ratio, if False: bckg corrected beforehand/ data bckgd less

    Returns
    -------
    indrec : position estimator in index coordinates (MLE)
    pos_estimator : position estimator (MLE)
    Ltot : Likelihood function
    
    Parameters 
    ----------
    step_nm : grid step in nm
        
    """

    step_nm = 1
       
    # number of beams in EBP
    K = np.shape(PSF)[0]
    # FOV size
    size = np.shape(PSF)[1] 
    
    normPSF = np.sum(PSF, axis = 0)
    
    # probabilitiy vector 
    p = np.zeros((K, size, size))
    
    if type(SBR) == bool: #bckg corrected elsewhere
        for i in np.arange(K):        
            p[i,:,:] = PSF[i,:,:]/normPSF 
    elif SBR_init == True: #normal case, PIE bckg correction
        for i in np.arange(K):
            p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * ratios[i]/np.sum(ratios)
    else: #normal case, CW bckg correction necessary
        for i in np.arange(K):        
            p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * (1/K)
            # p[i,:,:] = (1/SBR*(SBR + 1)) * (PSF[i,:,:]-  (1/(SBR + 1)) * (1/K)) /normPSF

                
    # likelihood function
    L = np.zeros((K,size, size))
    for i in np.arange(K):
        L[i, :, :] = n[i] * np.log(p[i, : , :])
        
    Ltot = np.sum(L, axis = 0)

    # maximum likelihood estimator for the position    
    indrec = np.unravel_index(np.argmax(Ltot, axis=None), Ltot.shape)
    indrec2 = [0,0]
    indrec2[0] = indrec[1]
    indrec2[1] = indrec[0]    
    pos_estimator = indexToSpace(indrec2, size, step_nm)
    
    return indrec2, pos_estimator


def fitgaussian(data,params):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    #    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = opt.leastsq(errorfunction, params) #xy transposed...
    pfit, pcov, infodict, errmsg, success = opt.leastsq(errorfunction, params, full_output=1)
    p = pfit
    print(p)
    print(np.sqrt(np.diag(pcov)))
    print(success)
    return p

def do_fit(xedges, yedges, H, params):

    newparams = []
    for gausparams in params:
    
        temp_parms = gausparams
        temp_parms[1] = temp_parms[1] - xedges[0]
        temp_parms[2] = temp_parms[2] - yedges[0]
        newparams.append(temp_parms)
    print(newparams)
    fitparams = fitgaussian(H, sum(newparams,[]))
    
    return fitparams

def gaussian(*params):
    """Returns a Gaussian function with the given parameters"""
    params = params
    return lambda x,y: sum([  params[5*i]*np.exp(-((x- params[5*i+1])**2/(2* params[5*i+3]**2)+(y- params[5*i+2])**2/(2* params[5*i+4]**2))) for i in np.arange(0,int(len(params)/5))])

def exgauss(x, var):
    '''convolution of Gaussian with exponential distribution, IRF fitting'''
    [mu, sig, tau] = var
    lam = 1/tau
    return 0.5*lam*np.exp(0.5*lam * (2*mu + lam*(sig**2) - 2*x))*ssp.erfc((mu + lam*(sig**2) - x)/(np.sqrt(2)*sig))
    

def ex2gauss(time, var, lifetime):
    lam0 = 1/var[2]
    lam1 = 1/ lifetime
    var1 = np.copy(var)
    var1[-1] = lifetime
    return -1*(lam0*lam1)/(lam0-lam1)*(1/lam0*exgauss(time, var) - 1/lam1*exgauss(time, var1))


def ex2gauss_generator(var, lifetime):
    '''generates convolution of Gaussian with two exponential distributions, 
    convolution of IRF with exponential decay of lifetime'''
    
    lam0 = 1/var[2]
    lam1 = 1/ lifetime
    var1 = np.copy(var)
    var1[-1] = lifetime

    def ex2gauss(time):
        return -1*(lam0*lam1)/(lam0-lam1)*(1/lam0*exgauss(time, var) - 1/lam1*exgauss(time, var1))
    return ex2gauss

def unbin_generator(function, data):
    '''creates function to be minimised in the MLE 
    from funciton which is to be fitted and the corresponding data'''
    def unbin_func(coeffs):
        if np.sum(coeffs[:3]) <= 1:
            y = function(data, coeffs)
            y = np.log(y)
            return -1 * np.sum(y)
        else:
            return np.inf
    return unbin_func

def MLE(fit_function, data, coeffs, limits):
    '''performs MLE, fitting the data to the fit function
    '''
    
    #create score function for MLE
    unbin = unbin_generator(fit_function, data)

    #run score function minimisation
    m = iminuit.Minuit.from_array_func(unbin, coeffs, limit = limits, print_level = 0, 
                                       error = [0.01]*len(coeffs), errordef = 0.5)
    m.migrad()
    params = m.values.values()
    errs = m.errors.values()
    
    return params, errs

def background_generator(data_IRF):
    '''creates function to fit fluorescent background biexponentially
    with IRF convolution'''
    def ex2gauss_background(time, coeffs):
        
        #order coefficients for fitting
        coeffs_exp4 = np.array(coeffs[:3])
        coeffs_exp4 = np.append(coeffs_exp4, 1 - np.sum(coeffs_exp4))
        coeffs_biexp = np.array(coeffs[3:7])
        coeffs_biexp = np.append(coeffs_biexp, 1 - coeffs_biexp)
        coeffs_biexp = coeffs_biexp.reshape((4,2), order = 'F')
        lifetimes = np.array(coeffs[7:9])
        
        intensity = np.zeros(time.shape)
        for i in np.arange(len(data_IRF)):
            for j in np.arange(len(lifetimes)):
                intensity += coeffs_exp4[i] * coeffs_biexp[i,j] * ex2gauss(time, data_IRF[i], lifetimes[j])
        return intensity
    return ex2gauss_background
            
def exponential_decay(time, time_0, lifetime):
    '''
    fit monoexponential decay starting from time_0 
    '''
    y = 1/lifetime*np.exp(-1/lifetime*(time-time_0))
    return y

def exp4(time, time_cycle, time_0s, coeffs_exp4, background, lifetime):
    'fit funciton for fitting epx4'
    # calculate exponential decays with heaviside function
    time_0s = np.array(time_0s).reshape((4,1))
    time = np.array([time]*4).reshape((4,-1))
    y = exponential_decay(time, time_0s, lifetime) * coeffs_exp4 * np.heaviside(time-time_0s, 1) 
    
    # add crosstalk by adding second cycle
    y += exponential_decay((time + time_cycle), time_0s, lifetime)*coeffs_exp4 
    y = np.sum(y, axis = 0)
    
    # add uniform background
    y = (1 - background)*y + background/time_cycle
    
    return y

def exp4_generator(time_0s,time_cycle):
    '''creates fit function for exp4 fitting with lifetime fitting included'''
    
    def exp4_func(time, coeffs):
        # reshapes coeffs
        coeff3 = 1 - np.sum(coeffs[:3])
        coeffs_exp4 = np.append(coeffs[:3], coeff3)
        coeffs_exp4 = coeffs_exp4.reshape((4,1))
        background = coeffs[3]
        lifetime = coeffs[4]
        
        y = exp4(time, time_cycle, time_0s, coeffs_exp4, background, lifetime)
        return y
    return exp4_func


def exp4_lifetime_fixed_generator(time_0s,time_cycle, lifetime):
    '''creates fit function for exp4 fitting with fixed lifetime'''
    def exp4_lifetime(time, coeffs):
        # reshapes coeffs
        coeff3 = 1 - np.sum(coeffs[:3])
        coeffs_exp4 = np.append(coeffs[:3], coeff3)
        coeffs_exp4 = coeffs_exp4.reshape((4,1))
        background = coeffs[3]
        
        y = exp4(time, time_cycle, time_0s, coeffs_exp4, background, lifetime)     
        return y    
    return exp4_lifetime


def exp4_IRF_generator(IRF_data, time_cycle, bckg_coeff,  bckg_func, bckg_params):
    '''creates fit function for exp4 fitting with IRF convolution and 
    lifetime fitting included'''
    def exp4_IRF(time, coeffs):
        #reshape coeffs
        coeffs_exp4 = np.array(coeffs[:3])
        coeffs_exp4 = np.append(coeffs_exp4, 1 - np.sum(coeffs_exp4))
        lifetime = coeffs[3]
        
        intensity = np.zeros(time.shape)
        for i in np.arange(len(IRF_data)):
            for j in np.arange(2): #adds crosstalk
                intensity += coeffs_exp4[i]  * ex2gauss(time + j*time_cycle, IRF_data[i], lifetime)
        intensity = (1-bckg_coeff)*intensity + bckg_coeff*bckg_func(time, bckg_params) #adds fluorescent background
        return intensity
    return exp4_IRF


def exp4_convolution(time, coeffs, conv, time_cycle):
    '''fits exp4 with IRF convolution, no background added'''
    #reshape coeffs for exp4_fitting
    coeff3 = 1 - np.sum(coeffs[:3])
    coeffs_exp4 = np.append(coeffs[:3], coeff3)
    coeffs_exp4 = coeffs_exp4.reshape((4,1))
    
    intensity = np.zeros(time.shape)
    for i in np.arange(len(conv)):
        intensity += conv[i,0](time)*coeffs_exp4[i]
        intensity += conv[i,0](time+time_cycle)*coeffs_exp4[i]
    return intensity


def exp4_IRF_lifetime_fixed_generator(conv, time_cycle):
    '''creates fit function for exp4 fitting with IRF convolution included and 
    priorly fixed lifetime, uniform background'''
    def exp4_IRF(time, coeffs):
        coeffs_exp4 = coeffs[:3]
        background = coeffs[3]
        intensity = exp4_convolution(time, coeffs_exp4, conv, time_cycle) #exp4 fitting
        intensity = (1-background)*intensity + background/time_cycle #adds uniform background
        return intensity
    return exp4_IRF


def exp4_IRF_lifetime_fixed_fluorescent_bckg_generator(conv, time_cycle, bckg_coeff, bckg_func, bckg_params):
    '''creates fit function for exp4 fitting with IRF convolution included, 
    priorly fixed lifetime and priorly determined fluorescent background'''
    def exp4_IRF_fluorescent(time, coeffs):
        coeffs_exp4 = coeffs[:3]
        intensity = exp4_convolution(time, coeffs_exp4, conv, time_cycle) #exp4 fitting
        intensity = (1-bckg_coeff)*intensity + bckg_coeff*bckg_func(time, bckg_params) #adds fluorescent background
        return intensity
    return exp4_IRF_fluorescent

def biexp_coeff_shaping(coeffs, percentages):
    #exp4 coeffs
    coeffs_exp4 = coeffs[:3]
    coeffs_exp4 = np.append(coeffs_exp4, 1 - np.sum(coeffs_exp4)).reshape((4,1))
    
    #biexp coeffs
    coeffs_biexp0 = np.array(coeffs[3:7]).reshape(4,1)
    coeffs_biexp1 = (1 - coeffs_biexp0)
    if len(percentages) == 3:
        coeffs_biexp0 = [coeffs_biexp0]*2
    else:
        coeffs_biexp0 = [coeffs_biexp0]
    coeffs_biexp = list(coeffs_biexp0)
    coeffs_biexp.append(coeffs_biexp1)
    coeffs_biexp = np.concatenate(coeffs_biexp, axis = 1)
    
    return coeffs_exp4, coeffs_biexp


def biexp_IRF_generator(time_cycle, conv, percentages, bckg_coeff, bckg_func, bckg_params):
    '''creates fit function for biexponential exp4 fitting with IRF convolution included, 
    priorly fixed lifetimes and uniform background'''

    def biexp_IRF(time, coeffs):
        coeffs_exp4, coeffs_biexp = biexp_coeff_shaping(coeffs, percentages)        
        intensity = np.zeros(time.shape)
        for i in np.arange(len(coeffs_exp4)):
            for j in np.arange(len(percentages)):
                intensity += conv[i,j](time)*percentages[j]*coeffs_biexp[i,j]*coeffs_exp4[i] #fluorescence
                intensity += conv[i,j](time+time_cycle)*percentages[j]*coeffs_biexp[i,j]*coeffs_exp4[i] #crosstalk    
        intensity = (1-bckg_coeff)*intensity + bckg_coeff*bckg_func(time, bckg_params) #adds fluorescent background
        return intensity
    return biexp_IRF  

def biexp_generator(time_0s,time_cycle, lifetimes_fixed, percentages):
    '''creates fit function for biexponential exp4 fitting with IRF convolution included, 
    priorly fixed lifetimes and uniform background'''
    def biexp(time, coeffs):
        coeffs_exp4, coeffs_biexp = biexp_coeff_shaping(coeffs, percentages)
        intensity = np.zeros(time.shape)
        for i in np.arange(len(coeffs_exp4)):
            for j in np.arange(len(percentages)):
                intensity += exponential_decay(time, time_0s[i], lifetimes_fixed[j])*coeffs_exp4[i]*percentages[j]*coeffs_biexp[i,j]*np.heaviside(time-time_0s[i],1)
                intensity += exponential_decay(time + time_cycle, time_0s[i], lifetimes_fixed[j])*coeffs_exp4[i]*percentages[j]*coeffs_biexp[i,j]*np.heaviside(time+time_cycle-time_0s[i],1)
                # add uniform background
        intensity = (1 - coeffs[7])*intensity + coeffs[7]/time_cycle
        
        return intensity
    return biexp

def unbin_generator_Poisson(function, data):
    '''creates function to be minimised in the MLE 
    from funciton which is to be fitted and the corresponding data'''
    def unbin_func(coeffs):
        #create binned data sets
        bins = 0.016 #bins in ns
        binning = np.arange(0, 51.3, bins)
        binned_times = binning[:-1] + bins/2
        y_data = np.histogram(data, bins = binning)[0]
        if np.sum(coeffs[:3]) <= 1:
            y_model = function(binned_times, coeffs)   
            return (y_model - y_data*np.log(y_model)).sum()
        else:
            return np.inf
    return unbin_func

def MLE_Poisson(fit_function, data, coeffs, limits):
    '''performs MLE, fitting the data to the fit function
    '''
    
    #create score function for MLE
    unbin = unbin_generator_Poisson(fit_function, data)

    #run score function minimisation
    m = iminuit.Minuit.from_array_func(unbin, coeffs, limit = limits, print_level = 0, 
                                       error = [0.01]*len(coeffs), errordef = 0.5)
    m.migrad()
    params = m.values.values()
    errs = m.errors.values()
    
    return params, errs