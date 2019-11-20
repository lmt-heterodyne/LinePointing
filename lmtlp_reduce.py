import sys
import numpy
import ast
from linepoint import linepoint
import matplotlib.pyplot as plt

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

    print('args = ', obsnum, opt, line_list, baseline_list, tsys)
    lp_file,params,ifproc_file_data,lp_stats_all = linepoint(obsnum, opt=opt, line_list=line_list, baseline_list=baseline_list, tsys=tsys)

    if lp_file is not None:
        status = 0
        x0 = 0
        x1 = 0
        x2 = 0
        if params is not None:
            print('params = ', params)
            l = len(params[0,:])
            if l > 0:
                x0 = numpy.mean(params[:,0])
            if l > 1:
                x1 = numpy.mean(params[:,1])
            if l > 2:
                x2 = numpy.mean(params[:,2])
        return status, x0, x1, x2, lp_file
    else:
        return -1, 0, 0, 0, None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: python3 lmtlp_reduce obsnum[;{key1}:{val1};{key2}:{val2};...]')
        sys.exit(-1)

    lmtlp_reduce(sys.argv[1])
    plt.show()
