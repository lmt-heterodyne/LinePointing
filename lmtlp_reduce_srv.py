import threading
from contextlib import closing
from socket import *
import time 
import re
import select
import sys
import traceback
from lmtlp_reduce import lmtlp_reduce
        
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
                status,x,y,pk,plotfile = self.manageCommand(msg)

                print (status, x, y, pk)
                if status == 0:
                    result = str.encode('0,{x:1.3f},{y:1.3f},{pk:1.6f}'.format(x=x, y=y, pk=pk))
                else :
                    result = b'-1,0,0,0'
                    
                conn.send(result)
                if plotfile is not None:
                    time.sleep(1)
                    with open(plotfile, 'rb') as f:
                        conn.sendfile(f, 0)
                break


            conn.close()

    def manageCommand(self, msg) :
        print (msg)
        try:
            return lmtlp_reduce(msg)
        except Exception as e:
            print (e)
            traceback.print_exc()
            return -1,0,0,0,None

def lmtlp_reduce_srv() :
    srv = LmtlpReduceSrv()
    srv.loop()

if __name__ == '__main__':
    lmtlp_reduce_srv()
