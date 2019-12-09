import sys
import socket
import subprocess
import os
import numpy as np
from lmtlp_reduce import lmtlp_reduce

def lmtlp_reduce_cli(host, port, obsnum, **args) :
    #
    # args -> string
    #

    argstr = ';'.join(['{key}:{val}'.format(key=x, val=args[x]) for x in args])
    msg = '{obsnum};{argstr}'.format(obsnum=obsnum,argstr=argstr)

    if host is None:
        status,x,y,pk,plotfile = lmtlp_reduce(msg)
        image_file = 'lmtlp_%s.png'%str(obsnum)
        print ('lmtlp_plotfile = ', plotfile)
        print ('lmtlp_imagefile = ', image_file)
        pcopy = subprocess.Popen(["cp", "-p", plotfile, image_file])
        sts = os.waitpid(pcopy.pid, 0)
        if status == 0:
            result = str.encode('0,{x:1.3f},{y:1.3f},{pk:1.6f}'.format(x=x, y=y, pk=pk))
        else :
            result = b'-1,0,0,0'
        return result


    s = socket.socket()
    s.connect((host, port))
    #s.send(('{};{}'.format(obsnum,argstr)).encode())
    #s.send(('%s;%s'%(obsnum,argstr)).encode())
    s.send(msg.encode())

    res = np.zeros(4)
    print ('recv size', res.itemsize*4)
    msg = s.recv(res.itemsize*4)
    res = np.frombuffer(msg)
    print ('msg', msg, len(msg))
    print ('res', res)
    #res = msg.decode().split(',')
    #print ('res', res)

    if res[0] == 0:
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

    ans = '{s:1.0f},{x:1.3f},{y:1.3f},{pk:1.6f}'.format(s=res[0],x=res[1], y=res[2], pk=res[3])
    return ans


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: python3 lmtlp_reduce_cli obsnum opt line_list baseline_list tsys')
        sys.exit(-1)

    msg = lmtlp_reduce_cli('wares', 16213, sys.argv[1], opt=sys.argv[2], line_list=sys.argv[3], baseline_list=sys.argv[4], tsys=sys.argv[5], tracking_beam=sys.argv[6])
    print (msg)
    
