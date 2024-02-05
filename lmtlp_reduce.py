import sys
import numpy
import ast
from linepoint import linepoint
import matplotlib.pyplot as plt
import json

def lmtlp_reduce(args_dict) :

    print ('args_dict = ', args_dict)


    lp_dict = linepoint(args_dict)
    plot_file = lp_dict.get('plot_file', None)
    params = lp_dict.get('params', None)
    ifproc_file_data = lp_dict.get('ifproc_data', None)

    results_dict = dict()
    if params is not None:
        status = 0
        x = 0
        y = 0
        pk = 0
        az_user_off = ifproc_file_data.az_user / 206264.8
        el_user_off = ifproc_file_data.el_user / 206264.8
        obspgm = ifproc_file_data.obspgm
        if params is not None:
            print('params = ', params)
            l = len(params[0,:])
            if l > 0:
                pk = numpy.mean(params[:,0])
            if l > 1:
                x = numpy.mean(params[:,1])
            if l > 2:
                y = numpy.mean(params[:,2])
        results_dict['x'] = x
        results_dict['y'] = y
        results_dict['pk'] = pk
        results_dict['az_user_off'] = ifproc_file_data.az_user / 206264.8
        results_dict['el_user_off'] = ifproc_file_data.el_user / 206264.8
        results_dict['obspgm'] = ifproc_file_data.obspgm
        try:
            for a in ['peak_fit_params', 'peak_fit_errors', 'peak_fit_snr', 'clipped', 'pixel_list']:
                obj = lp_dict[a]
                if isinstance(obj, (numpy.ndarray,)):
                    obj = obj.tolist()
                results_dict[a] = obj
        except Exception as e:
            print('lmtlp_reduce exception', e)
    else:
        status = -1
    results_dict['status'] = status
    results_dict['plot_file'] = plot_file
    results_str = json.dumps(results_dict)
    print('lmtlp_reduce results_str = ', results_str)
    return results_str

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: python3 lmtlp_reduce ObsNum')
        sys.exit(-1)

    args_dict = dict()
    args_dict['ObsNum'] = int(sys.argv[1])
    lmtlp_reduce(args_dict)
    plt.show()
