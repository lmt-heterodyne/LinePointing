import sys
import socket
import subprocess
import os
import numpy as np
from lmtlp_reduce import lmtlp_reduce
import json

def lmtlp_reduce_cli(host, port, args_dict) :
    print('lmtlp_reduce_cli args_dict = ', args_dict)
    obsnum = args_dict.get('ObsNum')

    if host is None:
        print('lmtlp_reduce_cli host is None, run directly')
        results_str = lmtlp_reduce(args_dict)
        results_dict = json.loads(results_str)
        status = results_dict['status']
        x =  results_dict['x']
        y =  results_dict['y']
        pk =  results_dict['pk']
        plotfile =  results_dict['plot_file']
        image_file = 'lmtlp_%s.png'%str(obsnum)
        print ('lmtlp_plotfile = ', plotfile)
        print ('lmtlp_imagefile = ', image_file)
        pcopy = subprocess.Popen(["cp", "-p", plotfile, image_file])
        sts = os.waitpid(pcopy.pid, 0)
        if status == 0:
            result = str.encode('0,{x:1.3f},{y:1.3f},{pk:1.6f}'.format(x=x, y=y, pk=pk))
        else :
            result = b'-1,0,0,0'
        return results_str


    s = socket.socket()
    s.connect((host, port))
    s.send(json.dumps(args_dict))

    res = np.zeros(1)
    print ('recv size', res.itemsize*1)
    msg = s.recv(res.itemsize*1)
    res = np.frombuffer(msg)
    print ('results_str len =', res[0])
    results_str = s.recv(int(res[0])).decode()
    print ('results_str =', results_str)
    results_dict = json.loads(results_str)
    status = results_dict['status']

    if status == 0:
        with open('lmtlp_%s.png'%obsnum, 'wb') as f:
            print ('image file opened')
            while True:
                data = s.recv(1024)
                if not data:
                    break
                # write data to a file
                f.write(data)
        f.close()
        print('image file closed')
        
    s.close()
    print('connection closed')

    return results_str


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: python3 lmtlp_reduce_cli ObsNum SpecOrCont LineList BaselineList BaselineFitOrder TSys TrackingBeam')
        sys.exit(-1)

    args_dict = dict()
    args_dict['ObsNum'] = sys.argv[1]
    args_dict['SpecOrCont'] = sys.argv[2]
    args_dict['LineList'] = sys.argv[3]
    args_dict['BaseLineList'] = sys.argv[4]
    args_dict['BaselineFitOrder'] = sys.argv[5]
    args_dict['TSys'] = sys.argv[6]
    args_dict['TrackingBeam'] = sys.argv[7]
    
    msg = lmtlp_reduce_cli('localhost', 16213, args_dict)
    print ('msg =', msg)
    
