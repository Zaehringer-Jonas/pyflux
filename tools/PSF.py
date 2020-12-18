# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:15:20 2018

@author: Luciano, Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

π = np.pi 


def convolved(grid, amplitudeG, amplitudeC, x0, y0, r, σ_x, σ_y, offset, theta=0):
    
    size = np.shape(grid[0])[0]
    
    c = circle(grid, amplitudeC, x0, y0, r)
    c = c.reshape(size, size)
    
    print(np.shape(c))
    
    g = gaussian2D(grid, amplitudeG, x0, y0, σ_x, σ_y, offset)
    g = g.reshape(size, size)
    
    print(np.shape(g))
    
    data = sg.fftconvolve(g, c)
    
    return data

def circle(grid, amplitude, x0, y0, r):
    
    n = np.shape(grid[0])[0]
    x, y = grid
#    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    mask = (x-x0)**2 + (y-y0)**2 <= r**2

    array = np.zeros((n, n))
    array[mask] = amplitude
    
    return array.ravel()

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
    
def doughnut2D(grid, A, x0, y0, d, offset):
    
    x, y = grid
    x0 = float(x0)
    y0 = float(y0) 
    r = np.sqrt((x-x0)**2 + (y-y0)**2) 
    
    D = offset + A * (r**2/d**4) * np.exp(-4 * np.log(2) * (r**2/d**2))
    
    return D.ravel()





def poly_func(grid, x0, y0, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
                  c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
                  c40, c41, c42, c43, c44):
        
        """    
        Polynomial function to fit PSFs.
        Uses built-in function polyval2d.
        
        Inputs
        ----------
        grid : x,y array
        cij : coefficients of the polynomial function
    
        Returns
        -------
        q : polynomial evaluated in grid.
        
        """
    
        x, y = grid
        c = np.array([[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14], 
                      [c20, c21, c22, c23, c24], [c30, c31, c32, c33, c34],
                      [c40, c41, c42, c43, c44]])
        q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)
    
        return q.ravel()

def poly_func2Dpl( x0, y0, c00, c01, c02, c10, c11, c12, 
                  c20, c21, c22):
        
        """    
        Polynomial function to fit PSFs.
        Uses built-in function polyval2d.
        
        Inputs
        ----------
        grid : x,y array
        cij : coefficients of the polynomial function
    
        Returns
        -------
        q : polynomial evaluated in grid.
        
        """
        def poly_local(grid):

            
            """    
            Polynomial function to fit PSFs.
            Uses built-in function polyval2d.
            
            Inputs
            ----------
            grid : x,y array
            cij : coefficients of the polynomial function
        
            Returns
            -------
            q : polynomial evaluated in grid.
            
            """
            print(grid)
            x, y = grid
            c = np.array([[c00, c01, c02], [c10, c11, c12], 
                          [c20, c21, c22]])
            q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)
        
            return q.ravel()
        
        return poly_local


def poly_func2D(grid, x0, y0, c00, c01, c02, c10, c11, c12, 
                  c20, c21, c22):
        
        """    
        Polynomial function to fit PSFs.
        Uses built-in function polyval2d.
        
        Inputs
        ----------
        grid : x,y array
        cij : coefficients of the polynomial function
    
        Returns
        -------
        q : polynomial evaluated in grid.
        
        """
    
        x, y = grid
        c = np.array([[c00, c01, c02], [c10, c11, c12], 
                      [c20, c21, c22]])
        q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)
    
        return q.ravel()





if __name__ == '__main__':    
    
    fwhm = 320 # in nm
    σ = fwhm/2.35
    
    size = 1000 # in nm
    px = 1 # in nm
    
    x = np.arange(-size/2, size/2, px)
    y = np.arange(-size/2, size/2, px) 
    
    [Mx, My] = np.meshgrid(x, y)
    
    ####################### gaussian ################################# 
    
    A_gaussian = 1/(2*π*σ**2) # normalized gaussian
    dataG = gaussian2D((Mx, My), A_gaussian, 100, 100, σ, σ, 0, 0)
    #print(np.shape(dataG))
    
    dataG = dataG.reshape(size, size)
    #print(np.shape(dataG))
    
    plt.imshow(dataG)
    
    ####################### doughnut #################################  
    
    plt.figure()
    
    A_doughnut = (16/π) * np.log(2)**2 # normalized doughnut
    dataD = doughnut2D((Mx, My), A_doughnut, 100, 100, 200, 0)
    dataD = dataD.reshape(size, size)
    
    plt.imshow(dataD)
    
    print(np.sum(dataD))
    
    ######################### circle ###############################    
    
    plt.figure()

    r = 150 # in nm
    dataC = circle((Mx, My), 1, 0, 0, r)
    dataC = dataC.reshape(size, size)    
    
    plt.imshow(dataC)
    

