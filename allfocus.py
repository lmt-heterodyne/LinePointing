import sys
import numpy
import math
import matplotlib.pyplot as pl
import time
import os
import numpy as np
from lmtslr.ifproc.ifproc import lookup_ifproc_file, IFProcData

from linepoint import linepoint
from m2fit import m2fit
from m2fit_viewer import m2fit_viewer
from merge_png import merge_png
from merge_focus import merge_focus

def allfocus(obsNums, peaks, lp_files, opt):

    # define time stamp
    obsnum = int(obsNums[-1])
    file_ts = '%d_%d_%d'%(obsnum, int(time.time()*1000), os.getpid())

    lp_params = []
    ifproc_file_data = []
    for i,obsnum in enumerate(obsNums):
        lp_params_1 = np.zeros((1,1))
        lp_params_1[0,0] = peaks[i]
        ifproc_file = lookup_ifproc_file(obsnum)
        if not ifproc_file:
            return -1
        ifproc_file_data_1 = IFProcData(ifproc_file)
        if ifproc_file_data_1 == None:
            return -1
        lp_params += [lp_params_1]
        ifproc_file_data += [ifproc_file_data_1]
    f = m2fit(lp_params,ifproc_file_data)
    f.find_focus()
    f.fit_focus_model()
    lp_p = [l[0][0] for l in lp_params]
    print 'lp_p', lp_p
    print 'relative_focus',f.relative_focus_fit
    print 'absolute_focus',f.absolute_focus_fit
    print 'm2 x y z',f.m2xfocus,f.m2yfocus,f.m2zfocus
    print 'm1 zer0',f.m1ZernikeC0

    params = numpy.zeros((1,4))
    params[0,0] = f.m2xfocus
    params[0,1] = f.m2yfocus
    params[0,2] = f.m2zfocus
    params[0,3] = f.m1ZernikeC0
    
    FV = m2fit_viewer()

    FV.set_figure(figure=100)
    FV.open_figure()
    FV.plot_fits(f,obsNums)

    pl.savefig('lf_fits_%s.png'%file_ts, bbox_inches='tight')

    FV.set_figure(figure=101)
    FV.open_figure()
    FV.plot_focus_model_fit(f,obsNums)
    pl.savefig('lf_model_%s.png'%file_ts, bbox_inches='tight')
    merge_png(['lf_fits_%s.png'%file_ts, 'lf_model_%s.png'%file_ts], 'lf_focus_%s.png'%file_ts)
    lp_files += ['lf_fits_%s.png'%file_ts]
    lp_files += ['lf_model_%s.png'%file_ts]
    lp_files += ['lf_focus_%s.png'%file_ts]
    merge_focus(lp_p, lp_files)

    if opt & 0x1:
        pl.show()
        
    print 'params', params
    return 'lf_focus_%s.png'%file_ts,params

if __name__ == '__main__':
    obsNums = [83578, 83579, 83580, 83581, 83582]
    peaks  = [8.386131542861701, 14.80816494349861, 16.462157307450138, 15.80190338676115, 9.86341460275963]
    imageFiles = ['lp_spec_83578_1568497923432_47231.png', 'lp_spec_83579_1568497924551_47231.png', 'lp_spec_83580_1568497924985_47231.png', 'lp_spec_83581_1568497925395_47231.png', 'lp_spec_83582_1568497925891_47231.png']
    allfocus(obsNums, peaks, imageFiles, 0x1)
    


