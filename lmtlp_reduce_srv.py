import threading
from contextlib import closing
from socket import *
import time 
import re
import select
import sys
import traceback
import numpy as np
from lmtlp_reduce import lmtlp_reduce
import json
        
class LmtlpReduceSrv :
    def __init__ (self) :
        self.listenPort = 16213
        self.bufsize    = 8192
        self.thisHost       = '0.0.0.0'
        self.debug = 0x0
        if(self.debug & 0x1): print("debug ", 0x1)

        #open listen port 
        print("listen at port %d" % self.listenPort)
        self.tcpServerSock = socket(AF_INET, SOCK_STREAM)
        self.tcpServerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR,1)
        self.tcpServerSock.bind((self.thisHost, self.listenPort))
        self.tcpServerSock.listen(10)
        
    def __del__ (self) :
        self.tcpServerSock.close()
        
    def loop(self) :
        while True :
            conn, addres = self.tcpServerSock.accept()
            if(self.debug & 0x1):
                print("accepted client")
            while True:
                msg = conn.recv(self.bufsize)
                if len(msg) == 0 :
                    if(self.debug & 0x1):
                        print('connection is closed by client')
                        print('closing socket...')
                    break

                print (msg)
                results_str = self.manageCommand(msg)
                results_dict = json.loads(results_str)

                status = results_dict['status']
                print (status)
                if status == 0:
                    plotfile = results_dict['plot_file']
                else :
                    plotfile = None

                res = np.zeros(1)
                res[0] = len(results_str)
                print ('results_str len =', res[0])
                print ('results_str =', results_str)
                conn.send(res.tobytes())
                conn.send(results_str.encode())
                if plotfile is not None:
                    try:
                        with open(plotfile, 'rb') as f:
                            conn.sendfile(f, 0)
                    except Exception as e:
                        print (e)
                break


            conn.close()

    def manageCommand(self, msg) :
        print (msg)
        try:
            return lmtlp_reduce(msg)
        except Exception as e:
            print (e)
            traceback.print_exc()
            results_dict = dict()
            results_dict['status'] = -1
            return json.dumps(results_dict)

def lmtlp_reduce_srv() :
    srv = LmtlpReduceSrv()
    srv.loop()

if __name__ == '__main__':
    lmtlp_reduce_srv()
