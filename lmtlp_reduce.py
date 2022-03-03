import sys
import numpy
import ast
from linepoint import linepoint
import matplotlib.pyplot as plt
import json

def lmtlp_reduce(msg) :

    #
    # msg => obsnum & args   : msg = '{obsnum};{key1}:{val1};{key2}:{val2}; ...'
    #

    print ('msg = ', msg)
    if isinstance(msg, str):
        argstrs = msg.split(';')
    else:
        argstrs = msg.decode().split(';')
    obsnum = int(argstrs[0])
    print ('argstrs = ', argstrs[1:])
    args = {}
    for x in argstrs[1:]:
        args[x[:x.find(':')]] = x[x.find(':')+1:]
    #args = { x[:x.find(':')] : x[x.find(':')+1:] for x in argstrs[1:] }
    print('args = ', args)

    opt      = int(args['opt']) if 'opt' in args else 0
    line_list = ast.literal_eval(args['line_list']) if 'line_list' in args else None
    baseline_list = ast.literal_eval(args['baseline_list']) if 'baseline_list' in args else None
    tsys = float(args['tsys']) if 'tsys' in args else None
    tracking_beam = ast.literal_eval(args['tracking_beam']) if 'tracking_beam' in args else None

    print('args = ', obsnum, opt, line_list, baseline_list, tsys, tracking_beam)
    plot_file,params,ifproc_file_data,lp_stats_all = linepoint(obsnum, opt=opt, line_list=line_list, baseline_list=baseline_list, tsys=tsys, tracking_beam=tracking_beam)

    results_dict = dict()
    if plot_file is not None:
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
        results_dict['status'] = status
        results_dict['x'] = x
        results_dict['y'] = y
        results_dict['pk'] = pk
        results_dict['plot_file'] = plot_file
        results_dict['az_user_off'] = ifproc_file_data.az_user / 206264.8
        results_dict['el_user_off'] = ifproc_file_data.el_user / 206264.8
        results_dict['obspgm'] = ifproc_file_data.obspgm
    else:
        results_dict['status'] = -1
    results_str = json.dumps(results_dict)
    print(results_str)
    return results_str

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: python3 lmtlp_reduce obsnum[;{key1}:{val1};{key2}:{val2};...]')
        sys.exit(-1)

    lmtlp_reduce(sys.argv[1])
    plt.show()
