from lmtslr.ifproc.ifproc import lookup_ifproc_file, IFProcQuick
import sys

def test(obsnum):
    print (obsnum)

    # read the ifproc file to get the data and the tracking beam
    ifproc_file = lookup_ifproc_file(obsnum)
    if not ifproc_file:
        return -1
    

    # probe the ifproc file for obspgm
    ifproc_file_quick = IFProcQuick(ifproc_file)

    # get obspgm
    obspgm = ifproc_file_quick.obspgm

    print ('obsnum', obsnum)
    print ('receiver', ifproc_file_quick.receiver)
    print ('obspgm', obspgm)

    return 0

            
if __name__ == '__main__':
    test(int(sys.argv[1]))
