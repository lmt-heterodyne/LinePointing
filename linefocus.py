import ast
import sys
from lmtlp_reduce_cli import lmtlp_reduce_cli
from allfocus import allfocus

def linefocus(obsNumList, opt, line_list, baseline_list, tsys, tracking_beam):
    obsNums = []
    peaks = []
    data_files = []
    image_files = []
    for i,obsN in enumerate(obsNumList):
      print('look for obsnum', obsN)
      msg = lmtlp_reduce_cli(None, 0, str(obsN), opt=opt, line_list=line_list, baseline_list=baseline_list, tsys=tsys, tracking_beam=tracking_beam)
      status,peak,azOff,elOff = msg.decode().split(',')
      if int(status) < 0:
        return False
      image_file = 'lmtlp_%s.png'%str(obsN)
      obsNums.append(obsN)
      peaks.append(peak)
      image_files.append(image_file)
    print(obsNums)
    print(peaks)
    print(image_files)
    imagefiles, params = allfocus(obsNums, peaks, image_files, opt)
    print(params)

if __name__ == '__main__':
    opt = 0
    if len(sys.argv) > 2:
        try:
            opt = int(sys.argv[1], 0)
            print ('opt =', opt)
        except:
            pass

    try:
        obsNumList = ast.literal_eval(sys.argv[-1])
    except Exception as e:
        print(e)
        print('must provide ObsNumList')
        sys.exit(0)
        
    linefocus(obsNumList, opt=opt, line_list=None, baseline_list=None, tsys=200, tracking_beam=None)

