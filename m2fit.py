import numpy
import math

class m2fit():
    """m2fit holds and analyzes variables for fitting intensities to derive an m2 or m1 parameter.
    """
    def __init__(self,lp_params,ifproc_file_data):
        """__init__ sets up parameters to hold data from linepoint parameters to derive an m2 or m1 parameter.
        
        input lp_params is an array of line point parameters
        input ifproc_file_data is an array of ifproc file data
        """
        # get common information 
        self.obsnum = ifproc_file_data[0].obsnum
        self.receiver = ifproc_file_data[0].receiver
        self.source = ifproc_file_data[0].source
        self.obsnum = ifproc_file_data[0].obsnum
        self.obspgm = ifproc_file_data[0].obspgm

        # determine x,y,z or zernike
        m2z = []
        m2y = []
        m2x = []
        m1zer0 = []
        for i,ifproc in enumerate(ifproc_file_data):
            if i != 0:
                if self.receiver != ifproc.receiver:
                    print 'receiver error'
                    return
                if self.source != ifproc.source:
                    print 'source error'
                    return
                if self.obspgm != ifproc.obspgm:
                    print 'obspgm error'
                    return
            m2z.append(ifproc.m2z)
            m2y.append(ifproc.m2y)
            m2x.append(ifproc.m2x)
            m1zer0.append(ifproc.m1ZernikeC0)
        
        m2z = numpy.array(m2z)
        m2y = numpy.array(m2y)
        m2x = numpy.array(m2x)
        m1zer0 = numpy.array(m1zer0)
        print 'm2x', m2x
        print 'm2y', m2y
        print 'm2z', m2z
        print 'm1zer0', m1zer0
        dx = max(m2x)-min(m2x)
        dy = max(m2y)-min(m2y)
        dz = max(m2z)-min(m2z)
        dzer = max(m1zer0)-min(m1zer0)
        self.m2xfocus = numpy.mean(m2x)
        self.m2yfocus = numpy.mean(m2y)
        self.m2zfocus = numpy.mean(m2z)
        self.m1ZernikeC0 = numpy.mean(m1zer0)

        if (dx == dy and dx == dz and dx == 0 and dzer == 0):
            #nothing's changing, an error should be thrown
            self.msg = "M2 or Zernike offsets are not changing in these files."
            m2pos = -1
        elif (dx != 0):
            if (dy != 0 or dz != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg = "More than one M2 offset is changing in these files."
                m2pos = -1
            else:
                m2pos = 2
        elif (dy != 0):
            if (dx != 0 or dz != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg = "More than one M2 or Zernike offset is changing in these files."
                m2pos = -1
            else:
                m2pos = 1
        elif (dz != 0):
            if (dx != 0 or dy != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg = "More than one M2 or Zernike offset is changing in these files."
                m2pos = -1
            else:
                m2pos = 0
        elif (dzer != 0):
            if (dx != 0 or dy != 0 or dz != 0):
                #more than one offset changing, throw an error
                self.msg = "More than one M2 or Zernike offset is changing in these files."
                m2pos = -1
            else:
                m2pos = 3

        self.m2pos = m2pos
        m2posLabel = {-1: 'Error', 0: 'Z', 1: 'Y', 2: 'X', 3: 'A'}
        print 'changing param:', m2posLabel[m2pos]

        self.nscans = len(lp_params)
        self.n = len(lp_params[0])

        self.data = numpy.zeros((self.nscans, self.n))
        for iscan in range(self.nscans):
            self.data[iscan] = lp_params[iscan][:,0]

        self.m2_position = numpy.zeros(self.nscans)
        self.m2_pcor = numpy.zeros(self.nscans)
        self.elev = numpy.zeros(self.nscans)
        for i,ifproc in enumerate(ifproc_file_data):
            if self.m2pos == 0:
                ave = ifproc.m2z
                pcor = ifproc.m2zPcor
            elif self.m2pos == 1:
                ave = ifproc.m2y
                pcor = ifproc.m2yPcor
            elif self.m2pos == 2:
                ave = ifproc.m2x
                pcor = ifproc.m2xPcor
            elif self.m2pos == 3:
                ave = ifproc.m1ZernikeC0
                pcor = 0
            else:
                ave = 0
                pcor = 0

            self.m2_position[i] = ave
            self.m2_pcor[i] = pcor
        self.parameters = numpy.zeros((self.n,3))
        self.result_relative = numpy.zeros(self.n)
        self.result_absolute = numpy.zeros(self.n)
    
    def find_focus(self):
        """Uses data loaded in during creation of this instance to fit focus."""
        if self.m2pos < 0: return
        for index in range(self.n):
            ptp = numpy.zeros((3,3))
            ptr = numpy.zeros(3)
            f = numpy.zeros(3)
            ee = []
            I = []
            par = []
            pcor = []
            for scan_id in range(self.nscans):
                I.append(self.data[scan_id][index])
                par.append(self.m2_position[scan_id])
                pcor.append(self.m2_pcor[scan_id])
                f[0] = 1.
                f[1] = par[scan_id]
                f[2] = par[scan_id]*par[scan_id]
                for ii in range(3):
                    for jj in range(3):
                        ptp[ii][jj] = ptp[ii][jj] + f[ii]*f[jj]
                    ptr[ii] = ptr[ii] + f[ii]*I[scan_id]
            ptpinv = numpy.linalg.inv(ptp)
            self.parameters[index,:] = numpy.dot(ptpinv,ptr)
            if self.parameters[index,2] != 0:
                self.result_relative[index] = -self.parameters[index,1]/self.parameters[index,2]/2.
                self.result_absolute[index] = self.result_relative[index] + numpy.mean(pcor)
            else:
                self.result_relative[index] = None
                self.result_absolute[index] = None


    def fit_focus_model(self):
        """Uses best fit focus (Z) for each instance to fit linear focus model."""
        if self.n > 1:
            xband = [index-0.5*(self.n-1) for index in range(self.n)]
            ptp = numpy.zeros((2,2))
            ptr = numpy.zeros(2)
            pta = numpy.zeros(2)
            f = numpy.zeros(2)
            for index in range(self.n):
                if(math.isnan(self.result_relative[index])):
                    continue
                f[0] = 1.
                f[1] = xband[index]
                for ii in range(2):
                    for jj in range(2):
                        ptp[ii][jj] = ptp[ii][jj] + f[ii]*f[jj]
                    ptr[ii] = ptr[ii] + f[ii]*self.result_relative[index]
                    pta[ii] = pta[ii] + f[ii]*self.result_absolute[index]
            ptpinv = numpy.linalg.inv(ptp)
            relative_focus_fit = numpy.dot(ptpinv,ptr)
            absolute_focus_fit = numpy.dot(ptpinv,pta)
            self.resids = numpy.zeros(self.n)
            resids_squared = 0.
            actual_n = 0
            for index in range(self.n):
                if(math.isnan(self.result_relative[index])):
                    continue
                self.resids[index] = self.result_relative[index] - relative_focus_fit[0] - relative_focus_fit[1]*xband[index]
                resids_squared = resids_squared + self.resids[index]*self.resids[index]
                actual_n = actual_n + 1
            rms = math.sqrt(resids_squared/actual_n)
            focus_error = math.sqrt(ptpinv[0][0])*rms

            self.relative_focus_fit = relative_focus_fit[0]
            self.focus_error = focus_error
            self.absolute_focus_fit = absolute_focus_fit[0]
            self.focus_slope = relative_focus_fit[1]
            self.fit_rms = rms
        elif self.n > 0:
            self.relative_focus_fit = self.result_relative[0]
            self.focus_error = 0
            self.absolute_focus_fit = self.result_absolute[0]
            self.focus_slope = 0
            self.fit_rms = 0
        if self.m2pos == 0:
            self.m2zfocus = self.relative_focus_fit
        elif self.m2pos == 1:
            self.m2yfocus = self.relative_focus_fit
        elif self.m2pos == 2:
            self.m2xfocus = self.relative_focus_fit
        elif self.m2pos == 3:
            self.m1ZernikeC0 = self.relative_focus_fit

