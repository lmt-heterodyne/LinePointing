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
        m2tip = []
        m2tilt = []
        m1zer0 = []
        self.status = []
        self.msg = []
        for i,ifproc in enumerate(ifproc_file_data):
            if i != 0:
                if self.receiver != ifproc.receiver:
                    self.msg.append('Receiver mismatch %d:%s %d:%s'%(self.obsnum, self.receiver, ifproc.obsnum, ifproc.receiver))
                    print(self.msg)
                    self.status.append(-1)
                    return
                if self.source != ifproc.source:
                    self.msg.append('Source mismatch %d:%s %d:%s'%(self.obsnum, self.source, ifproc.obsnum, ifproc.source))
                    print(self.msg)
                    self.status.append(-1)
                    return
                if self.obspgm != ifproc.obspgm:
                    self.msg.append('ObsPgm mismatch %d:%s %d:%s'%(self.obsnum, self.obspgm, ifproc.obsnum, ifproc.obspgm))
                    print(self.msg)
                    self.status.append(-1)
                    return
            m2z.append(ifproc.m2z)
            m2y.append(ifproc.m2y)
            m2x.append(ifproc.m2x)
            m2tip.append(ifproc.m2tip)
            m2tilt.append(ifproc.m2tilt)
            m1zer0.append(ifproc.m1ZernikeC0)
        
        m2z = numpy.array(m2z)
        m2y = numpy.array(m2y)
        m2x = numpy.array(m2x)
        m2tip = numpy.array(m2tip)
        m2tilt = numpy.array(m2tilt)
        m1zer0 = numpy.array(m1zer0)
        print('m2x', m2x)
        print('m2y', m2y)
        print('m2z', m2z)
        print('m2tip', m2tip)
        print('m2tilt', m2tilt)
        print('m1zer0', m1zer0)
        dx = max(m2x)-min(m2x)
        dy = max(m2y)-min(m2y)
        dz = max(m2z)-min(m2z)
        dtip = max(m2tip)-min(m2tip)
        dtilt = max(m2tilt)-min(m2tilt)
        dzer = max(m1zer0)-min(m1zer0)
        self.m2xfocus = numpy.mean(m2x)
        self.m2yfocus = numpy.mean(m2y)
        self.m2zfocus = numpy.mean(m2z)
        self.m2tipfocus = numpy.mean(m2tip)
        self.m2tiltfocus = numpy.mean(m2tilt)
        self.m1ZernikeC0 = numpy.mean(m1zer0)

        if (dx == 0 and dy == 0 and dz == 0 and dtip == 0 and dtilt == 0 and dzer == 0):
            #nothing's changing, an error should be thrown
            self.msg.append("M2 or Zernike offsets are not changing in these files.")
            m2pos = -1
        elif (dx != 0):
            if (dy != 0 or dz != 0 or dtip != 0 or dtilt != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 2
        elif (dy != 0):
            if (dx != 0 or dz != 0 or dtip != 0 or dtilt != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 or Zernike offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 1
        elif (dz != 0):
            if (dx != 0 or dy != 0 or dtip != 0 or dtilt != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 or Zernike offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 0
        elif (dtip != 0):
            if (dx != 0 or dy != 0 or dz != 0 or dtilt != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 or Zernike offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 4
        elif (dtilt != 0):
            if (dx != 0 or dy != 0 or dz != 0 or dtip != 0 or dzer != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 or Zernike offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 5
        elif (dzer != 0):
            if (dx != 0 or dy != 0 or dz != 0 or dtip != 0 or dtilt != 0):
                #more than one offset changing, throw an error
                self.msg.append("More than one M2 or Zernike offset is changing in these files.")
                m2pos = -1
            else:
                m2pos = 3

        self.m2pos = m2pos
        m2posLabel = {-1: 'Error', 0: 'Z', 1: 'Y', 2: 'X', 3: 'A', 4: 'Tip', 5: 'Tilt'}
        print('changing param:', m2posLabel[m2pos])

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
            elif self.m2pos == 4:
                ave = ifproc.m2tip
                pcor = ifproc.m2tipPcor
            elif self.m2pos == 5:
                ave = ifproc.m2tilt
                pcor = ifproc.m2tiltPcor
            else:
                ave = 0
                pcor = 0

            self.m2_position[i] = ave
            self.m2_pcor[i] = pcor
        self.parameters = numpy.zeros((self.n,3))
        self.result_relative = numpy.zeros(self.n)
        self.result_absolute = numpy.zeros(self.n)
        self.scans_xpos = []
        self.scans_xpos_all = []
    
    def find_focus(self, use_gaus=False):
        """Uses data loaded in during creation of this instance to fit focus."""
        if self.m2pos < 0: return

        mdata_max = numpy.amax(self.data, axis=0)
        print('data', self.data)
        print('n', self.n)
        print(mdata_max)
        for index in range(self.n):
            ptp = numpy.zeros((3,3))
            ptr = numpy.zeros(3)
            f = numpy.zeros(3)
            ee = []
            I = []
            par = []
            pcor = []
            print('index, mdata_max', index, mdata_max[index])
            scan_id_good = 0
            for scan_id in range(self.nscans):
                self.scans_xpos_all.append(self.m2_position[scan_id])
                print('scan_id, mdata, half max', scan_id, self.data[scan_id][index], 0.5*mdata_max[index])
                if use_gaus == False and self.data[scan_id][index] < 0.5*mdata_max[index]:
                    continue
                if self.data[scan_id][index] == 0:
                    continue
                I.append(self.data[scan_id][index])
                par.append(self.m2_position[scan_id])
                self.scans_xpos.append(self.m2_position[scan_id])
                pcor.append(self.m2_pcor[scan_id])
                f[0] = 1.
                f[1] = par[scan_id_good]
                f[2] = par[scan_id_good]*par[scan_id_good]
                for ii in range(3):
                    for jj in range(3):
                        ptp[ii][jj] = ptp[ii][jj] + f[ii]*f[jj]
                    ptr[ii] = ptr[ii] + f[ii]*I[scan_id_good]
                scan_id_good += 1
            print('I', I)
            print('par', par)
            print('xpos', self.scans_xpos)
            print('pcor', pcor)
            if len(I) <= 2 or len(set(par)) <= 2:
                self.result_relative[index] = 0
                self.result_absolute[index] = 0
                self.msg.append("Only %d data points are above half max"%len(I))
                self.status.append(-1)
                print('------------', self.msg)
            else:
                ptpinv = numpy.linalg.inv(ptp)
                self.parameters[index,:] = numpy.dot(ptpinv,ptr)
                if use_gaus == True:
                    from scipy.optimize import curve_fit
                    def gaus(x,a,x0,sigma):
                        return a*numpy.exp(-(x-x0)**2/(2*sigma**2))
                    I = numpy.array(I)
                    par = numpy.array(par)
                    ymean = max(I)
                    mean = sum(par*I)/sum(I)
                    sigma = numpy.sqrt(abs(sum((par-mean)**2*I)/sum(I)))
                    p0 = [ymean, mean, sigma]
                    print('gaus p0', p0)
                    popt,pcov = curve_fit(gaus,par,I,p0)
                    print('gaus popt ', popt)
                    self.result_relative[index] = popt[1]
                    self.parameters[index,0] = popt[0]
                    self.parameters[index,1] = popt[1]
                    self.parameters[index,2] = popt[2]
                elif self.parameters[index,2] != 0:
                    self.result_relative[index] = -self.parameters[index,1]/self.parameters[index,2]/2.
                else:
                    self.result_relative[index] = 0
                    self.result_absolute[index] = 0
                    self.msg.append("Problem in fit")
                    print(self.msg)
                    self.status.append(-1)
                self.result_absolute[index] = self.result_relative[index] + numpy.mean(pcor)


    def fit_focus_model(self, col_id=None, masks=None):
        """Uses best fit focus (Z) for each instance to fit linear focus model."""
        if self.receiver == 'RedshiftReceiver':
            xbands = set([int(col_id[0][index]) for index in range(self.n)])
            xband = [-1,-.2,-.6,.2,1.,.6]
        else:
            xbands = [index for index in range(self.n)]
            xband = [index-0.5*(self.n-1) for index in range(self.n)]
        print('n =', self.n)
        print('xbands =', xband)
        if self.n > 1 and len(xbands) > 1:
            print('fit focus')
            ptp = numpy.zeros((2,2))
            ptr = numpy.zeros(2)
            pta = numpy.zeros(2)
            f = numpy.zeros(2)
            result_median = numpy.median(self.result_relative)
            result_cutoff = 2.0 * numpy.std(self.result_relative)
            for index in range(self.n):
                if math.isnan(self.result_relative[index]) or abs(self.result_relative[index] - result_median) >= result_cutoff:
                    print('reject_focus_model', index, self.result_relative[index])
                    continue
                print('fit_focus_model', index, self.result_relative[index])
                f[0] = 1.
                if self.receiver == 'RedshiftReceiver':
                    f[1] = xband[int(col_id[0][index])]
                else:
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
                if self.receiver == 'RedshiftReceiver':
                    self.resids[index] = self.result_relative[index] - relative_focus_fit[0] - relative_focus_fit[1]*xband[int(col_id[0][index])]
                else:
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
            print('average focus')
            self.relative_focus_fit = numpy.mean(self.result_relative)
            self.focus_error = 0
            self.absolute_focus_fit = numpy.mean(self.result_absolute)
            self.focus_slope = 0
            self.fit_rms = 0
        if type(masks) == list and type(masks[0]) == list:
            print('has masks')
            if len(set(masks[0])) > 1:
                for i,m in enumerate(masks[0]):
                    if m == 1:
                        print('use result', i)
                        self.relative_focus_fit = self.result_relative[i]
                        self.focus_error = 0
                        self.absolute_focus_fit = self.result_absolute[i]
                        self.focus_slope = 0
                        self.fit_rms = 0
                        break
                
        if self.m2pos == 0:
            self.m2zfocus = self.relative_focus_fit
        elif self.m2pos == 1:
            self.m2yfocus = self.relative_focus_fit
        elif self.m2pos == 2:
            self.m2xfocus = self.relative_focus_fit
        elif self.m2pos == 3:
            self.m1ZernikeC0 = self.relative_focus_fit
        elif self.m2pos == 4:
            self.m2tipfocus = self.relative_focus_fit
        elif self.m2pos == 5:
            self.m2tiltfocus = self.relative_focus_fit

