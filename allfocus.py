import sys
import numpy
import math
import time
import os
import numpy as np
from lmtslr.ifproc.ifproc import IFProcData, IFProcQuick
from lmtslr.utils.ifproc_file_utils import lookup_ifproc_file_all
from msg_image import mkMsgImage
import glob
import matplotlib.pyplot as pl

from linepoint import linepoint
from m2fit import m2fit
from m2fit_viewer import m2fit_viewer
from merge_png import merge_png
from merge_focus import merge_focus

def allfocus(allfocus_list, opt=0):
    obsNums = [d.get('obsnum', -1) for d in allfocus_list]
    peaks = [d.get('peak', 0) for d in allfocus_list]
    image_files = [d.get('image_file', None) for d in allfocus_list]
    file_data = [d.get('file_data', None) for d in allfocus_list]
    row_id = [d.get('chassis_id', 0) for d in allfocus_list]
    col_id = [d.get('board_id', 0) for d in allfocus_list]
    masks = [d.get('mask', None) for d in allfocus_list]
    names = [d.get('name', None) for d in allfocus_list]

    print('===================================')
    print('allfocus')
    print('obsnums', obsNums)
    print('peaks', peaks)
    print('image_files', image_files)
    print('row_id', row_id)
    print('col_id', col_id)
    print('masks', masks)
    print('names', names)
    print('opt', opt)
    allfocus_results_d = {}
    allfocus_results_d['obsnum'] = obsNums
    allfocus_results_d['image_files'] = image_files
    

    # define time stamp
    file_ts = '%d_%d_%d'%(max(obsNums), int(time.time()*1000), os.getpid())

    lp_merge_image_files = []
    lp_merge_params = []
    lp_params = []
    ifproc_file_data = []
    for i,obsnum in enumerate(obsNums):
        if isinstance(peaks[i], list):
            print('peaks', peaks[i], len(peaks[i]))
            lp_params_1 = np.zeros((len(peaks[i]),1))
            for j in range(len(peaks[i])):
                lp_params_1[j,0] = peaks[i][j]
        else:
            lp_params_1 = np.zeros((1,1))
            lp_params_1[0,0] = peaks[i]
        ifproc_file_data_1 = file_data[i]
        if ifproc_file_data_1.obspgm == 'Bs' or ifproc_file_data_1.obspgm == 'Ps' or ifproc_file_data_1.obspgm == 'Map' or ifproc_file_data_1.obspgm == 'Lissajous':
            lp_merge_params += [float(np.mean(peaks[i]))]
        else:
            lp_merge_params += [1]
        lp_merge_image_files += [image_files[i]]
        lp_params += [lp_params_1]
        ifproc_file_data += [ifproc_file_data_1]
    f = m2fit(lp_params,ifproc_file_data)
    print('f.status', f.status)
    if (len(f.status) == 1 and f.status[0] < 0) or f.m2pos == -1:
        print(obsnum, f.msg)
        mkMsgImage(pl, obsnum, txt=f.msg[0], im='lf_focus_%s.png'%file_ts, label='Error', color='r')
        params = None
        allfocus_results_d['png'] = 'lf_focus_%s.png'%file_ts
        allfocus_results_d['sorted_image_files'] = []
        allfocus_results_d['params'] = params
        allfocus_results_d['status'] = f.status
        allfocus_results_d['message'] = f.msg
        allfocus_results_d['par'] = None
        allfocus_results_d['intensity'] = None
        allfocus_results_d['pos'] = 0
        return allfocus_results_d
        #return 'lf_focus_%s.png'%file_ts,params,f.status,f.msg

    use_gaus = False
    f.find_focus(use_gaus=use_gaus)
    f.fit_focus_model(col_id=col_id, masks=masks)
    print('lp_params', lp_params)
    print('relative_focus',f.relative_focus_fit)
    print('absolute_focus',f.absolute_focus_fit)
    print('m2 x y z tip tilt',f.m2xfocus,f.m2yfocus,f.m2zfocus,f.m2tipfocus,f.m2tiltfocus)
    print('m1 zer0',f.m1ZernikeC0)
    print('x ', f.scans_xpos_all)
    lp_merge_params = [i for _,i in sorted(zip(f.scans_xpos_all,lp_merge_params))]
    sorted_image_files =  [i for _,i in sorted(zip(f.scans_xpos_all,lp_merge_image_files))]
    lp_merge_image_files = [i for _,i in sorted(zip(f.scans_xpos_all,lp_merge_image_files))]

    params = numpy.zeros((1,6))
    params[0,0] = f.m2xfocus
    params[0,1] = f.m2yfocus
    params[0,2] = f.m2zfocus
    params[0,3] = f.m1ZernikeC0
    params[0,4] = f.m2tipfocus
    params[0,5] = f.m2tiltfocus
    print('params 1', params)

    print('open m2fit_viewer')
    FV = m2fit_viewer()

    print('set figure')
    FV.set_figure(figure=100)
    print('open figure')
    FV.open_figure()
    print('plot fits')
    FV.plot_fits(f,obsNums,use_gaus=use_gaus,row_id=row_id,col_id=col_id,names=names)
    print('save fits')
    FV.save_figure('lf_fits_%s.png'%file_ts)
    print('close fits')
    FV.close_figure(figure=100)

    print('set figure')
    FV.set_figure(figure=101)
    print('open figure')
    FV.open_figure()
    print('plot model')
    FV.plot_focus_model_fit(f,obsNums,row_id=row_id,col_id=col_id,names=names)
    print('save model')
    FV.save_figure('lf_model_%s.png'%file_ts)
    print('close model')
    FV.close_figure(figure=101)
    
    print('merge')
    merge_png(['lf_fits_%s.png'%file_ts, 'lf_model_%s.png'%file_ts], 'lf_focus_%s.png'%file_ts)
    lp_merge_image_files += ['lf_fits_%s.png'%file_ts]
    lp_merge_image_files += ['lf_model_%s.png'%file_ts]
    lp_merge_image_files += ['lf_focus_%s.png'%file_ts]
    if f.receiver != 'RedshiftReceiver' and f.receiver != 'Toltec':
        merge_focus(lp_merge_params, lp_merge_image_files)

    if opt & 0x1:
        FV.show()
        
    print('params', params)
    allfocus_results_d['png'] = 'lf_focus_%s.png'%file_ts
    allfocus_results_d['sorted_image_files'] = sorted_image_files
    allfocus_results_d['params'] = params
    allfocus_results_d['status'] = f.status
    allfocus_results_d['message'] = f.msg
    allfocus_results_d['par'] = [i for i in sorted(f.scans_xpos_all)]
    allfocus_results_d['intensity'] = lp_merge_params
    allfocus_results_d['pos'] = f.m2pos
    return allfocus_results_d

if __name__ == '__main__':
    obsNums = [83578, 83579, 83580, 83581, 83582]
    peaks  = [8.386131542861701, 14.80816494349861, 16.462157307450138, 15.80190338676115, 9.86341460275963]
    imageFiles = ['lp_spec_83578_1568497923432_47231.png', 'lp_spec_83579_1568497924551_47231.png', 'lp_spec_83580_1568497924985_47231.png', 'lp_spec_83581_1568497925395_47231.png', 'lp_spec_83582_1568497925891_47231.png']
    opt = 0x1
    obsNums = [93164, 93165, 93166, 93167, 93168]
    peaks = [1050.5671629157275, 1159.8579761398939, 1218.4598409124826, 1157.5620061643654, 1043.9765320135932]
    imageFiles = ['lmtlp_93164_*.png', 'lmtlp_93165_*.png', 'lmtlp_93166_*.png', 'lmtlp_93167_*.png', 'lmtlp_93168_*.png']
    for i,f in enumerate(imageFiles):
        for f1 in glob.glob(f):
            print(f1)
            imageFiles[i] = f1
    opt = 0
    allfocus(obsNums, peaks, imageFiles, opt)
    


