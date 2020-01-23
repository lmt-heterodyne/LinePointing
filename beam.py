""" Module for analysis of beam maps 
"""

import numpy as np
import math
from scipy.optimize import leastsq
from lmtslr.grid.grid import *
import sys

def fit_array_function(param,xdata):
    """ compute model function for grid fitting
        param is array with fit parameters: [azoffset, eloffset, spacing, angle]
        xdata is array with independent variable: [array_of_az_positions array_of_el_positions] 
    """
    n = len(xdata)
    x = math.pi/180.
    xmodel =  math.cos(x*param[3])*param[2]*xdata[:n/2]-math.sin(x*param[3])*param[2]*xdata[n/2:]+param[0]
    ymodel =  math.sin(x*param[3])*param[2]*xdata[:n/2]+math.cos(x*param[3])*param[2]*xdata[n/2:]+param[1]
    return(np.concatenate((xmodel,ymodel),0))

def compute_array_residuals(param,xdata,data):
    """ compute array of residuals to a particular model for grid fitting
        param is array with fit parameters: [azoffset, eloffset, spacing, angle]
        xdata is array with independent variable: [array_of_az_positions array_of_el_positions] 
        data is array with positions to be fit: [array_of_az_positions array_of_el_positions] 
    """
    model = fit_array_function(param,xdata)
    residuals = data - model
    return(residuals)

def compute_model(v,xdata,ydata):
    """computes gaussian 2d model from x,y; added 3/15/18 for least squares fit to beam
       v is array with gaussian beam paramters: [peak, azoff, az_hpbw, eloff, el_hpbw]
       xdata is array with x positions
       ydata is array with y positions
    """
    model = v[0]*np.exp(-4.*np.log(2.)*((xdata-v[1])**2/v[2]**2+(ydata-v[3])**2/v[4]**2))
    return(model)

def compute_the_residuals(v,xdata,ydata,data):
    """computes residuals to gaussian 2d model; added 3/15/18 for least squares fit to beam
       v is array with gaussian beam paramters: [peak, azoff, az_hpbw, eloff, el_hpbw]
       xdata is array with x positions
       ydata is array with y positions
       data is array with map values to be fit
    """
    n = len(data)
    model = compute_model(v,xdata,ydata)
    residuals = data-model
    return(residuals)

class BeamMap():
    def __init__(self,BData,pix_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        ''' creates object with data and methods for pointing fits to a grid
            BData    is the object with map data; the arrays map_x,map_y,map_data must
                     have been created for all the pixels you wish to analyze.
            pix_list is list of pixel id's to be analyzed
        '''
        self.BData = BData
        self.npix = BData.npix
        self.elev = BData.elev
        self.tracking_beam = BData.tracking_beam
        self.obsnum = BData.obsnum
        self.cal_flag = BData.cal_flag

        self.pix_list = np.array(pix_list)
        self.n_pix_list = len(self.pix_list)
        # grid locations by pixel number
        if BData.receiver == "Msip1mm":
            self.RIGHT = np.array([0, 0, 0, 0])
            self.UP = np.array([0, 0, 0, 0])
        else:
            self.RIGHT = np.array([-1.5, -1.5, -1.5, -1.5, -.5, -.5, -.5, -.5, .5, .5, .5, .5, 1.5, 1.5, 1.5, 1.5])
            self.UP = np.array([1.5, .5, -.5, -1.5, 1.5, .5, -.5, -1.5, 1.5, .5, -.5, -1.5, 1.5, .5, -.5, -1.5])

    def find_pixel_index(self,ipix):
        """ locates the index of a specific pixel in the "pix_list" of this BeamMap
        """
        index = np.where(self.pix_list == ipix)
        return(int(index[0]))

    def fit_peak(self,ipix,fit_circle=300):
        """ fits a gaussian to a single peak; ipix is the pixel id in the array
        """
        index = self.BData.find_map_pixel_index(ipix)
        theMax = np.max(self.BData.map_data[index][:])
        peak_index = np.where(self.BData.map_data[index][:] == theMax)
        xp = self.BData.map_x[index][peak_index][0]
        yp = self.BData.map_y[index][peak_index][0]
        print('found_peak',ipix,index,theMax,peak_index[0][0],xp,yp,theMax)
        if self.BData.receiver == "Msip1mm":
            hpbw = 6.
        else:
            hpbw = 15.
        v0 = np.array([theMax,xp,hpbw,yp,hpbw])
        spec_list = np.where(np.sqrt((self.BData.map_x[index]-xp)**2 + (self.BData.map_y[index]-yp)**2) < fit_circle)
        #print(spec_list)
        scan = self.BData.map_data[index][:]
        #for i in spec_list:
            #print(i,self.BData.map_x[index][i],self.BData.map_y[index][i],scan[i])
        lsq_fit,lsq_cov,lsq_inf,lsq_msg,lsq_success = leastsq(compute_the_residuals,
                                                              v0,
                                                              args=(self.BData.map_x[index][spec_list],
                                                                    self.BData.map_y[index][spec_list],
                                                                    scan[spec_list]),
                                                              full_output=1)
        residuals = compute_the_residuals(lsq_fit,self.BData.map_x[index][spec_list],self.BData.map_y[index][spec_list],scan[spec_list])
        chisq = np.dot(residuals.transpose(),residuals)
        npts = self.BData.map_n[index]
        print('peak fit chisq = %f     rms = %f'%(chisq,np.sqrt(chisq/npts)))
        #print 'lsq_fit',lsq_fit
        #print 'lsq_cov',lsq_cov
        #print 'lsq_msg',lsq_msg
        #print 'lsq_success',lsq_success
        if lsq_cov is None:
            lsq_err = 0
        else:
            lsq_err = np.sqrt(np.diag(lsq_cov)*chisq/(npts-5))
        if lsq_success > 4:
            lsq_status = False
        else:
            lsq_status = True
        return(lsq_fit,lsq_err,lsq_status,chisq)

    def compute_model(self,ipix,params):
        """ computes a model scan for the fit to a single peak
            ipix is the pixel id
            params is the gaussian parameters determined from a fit
        """
        index = self.BData.find_map_pixel_index(ipix)
        return compute_model(params,self.BData.map_x[index][:],self.BData.map_y[index][:])

    def compute_residuals(self,ipix,params):
        """ computes scan of residuals to a model fit
            ipix is the pixel id
            params is the gaussian parameters determined from a fit
        """
        index = self.BData.find_map_pixel_index(ipix)
        scan = self.BData.map_data[index][:]        
        return compute_the_residuals(params,self.BData.map_x[index][:],self.BData.map_y[index][:],scan)

    def fit_peaks_in_list(self,fit_circle=300):
        """ find all the peaks in the list of pixels given by pix_list
        """
        self.peak_fit_params = np.zeros((self.n_pix_list,5))
        self.peak_fit_errors = np.zeros((self.n_pix_list,5))
        self.peak_fit_status = np.zeros((self.n_pix_list))
        self.peak_fit_chisq = np.zeros((self.n_pix_list))
        self.model = []
        for i in range(self.n_pix_list):
            """ note that i is index to the pixels in the pix_list """
            ipix = self.pix_list[i]
            self.peak_fit_params[i,:],self.peak_fit_errors[i,:],self.peak_fit_status[i],self.peak_fit_chisq[i] = self.fit_peak(ipix,fit_circle)
            self.model.append(self.compute_model(ipix,self.peak_fit_params[i,:]))
            
    def fit_grid(self):
        """ fits the grid parameters for the array given a list of gaussian fit results
        """
        #format data and fit
        xdata = np.concatenate((self.RIGHT[self.pix_list],self.UP[self.pix_list]),0)
        data = np.concatenate((self.peak_fit_params[:,1],self.peak_fit_params[:,3]),0)
        param0=np.array([0,0,28,0])
        lsq_fit,lsq_cov,lsq_inf,lsq_err,lsq_success = leastsq(compute_array_residuals,param0,args=(xdata,data),full_output=1)
        self.grid_param = lsq_fit
        # compute residuals to the grid fit
        resids = data - fit_array_function(self.grid_param,xdata);
        self.grid_rms = np.sqrt(np.sum(resids**2)/2/self.n_pix_list);
        self.grid_param_error  = np.sqrt(np.diag(lsq_cov)*np.sum(resids**2)/(2.*self.n_pix_list-4.))
        
        
