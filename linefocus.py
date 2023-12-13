import ast
import sys
import json
from lmtlp_reduce_cli import lmtlp_reduce_cli
from lmtslr.ifproc.ifproc import lookup_ifproc_file, IFProcQuick, IFProcData, IFProcCal
from allfocus import allfocus

def linefocus(obsNumList, opt, line_list, baseline_list, baseline_fit_order, tsys, tracking_beam):
    obsNums = []
    peaks = []
    file_data = []
    image_files = []
    for i,obsN in enumerate(obsNumList):
      print('look for obsnum', obsN)
      args_dict = dict()
      args_dict['ObsNum'] = obsN
      args_dict['SpecOrCont'] = 'Cont' if opt & 0x1000 else 'Spec'
      args_dict['LineList'] = line_list
      args_dict['BaselineList'] = baseline_list
      args_dict['BaselineFitOrder'] = baseline_fit_order
      args_dict['TSys'] = tsys
      args_dict['TrackingBeam'] = tracking_beam
      args_dict['Opt'] = opt
      results_str = lmtlp_reduce_cli(None, 0, args_dict)
      print(results_str)
      results_dict = json.loads(results_str)
      status = results_dict['status']
      if status < 0:
          image_file = 'lmtlp_%s.png'%str(obsN)
          return None, None, status, 'Error from pointing reduction'
      peak = results_dict['pk']
      azOff = results_dict['x']
      elOff = results_dict['y']
      azUserOff = results_dict['az_user_off']
      elUserOff = results_dict['el_user_off']
      obsPgm = results_dict['obspgm']
      if int(status) < 0:
        return False
      image_file = 'lmtlp_%s.png'%str(obsN)
      # read the data file to get the data
      ifproc_file = lookup_ifproc_file(obsN)
      if not ifproc_file:
          print('no data file found for', obsN)
          return False
      file_data_1 = IFProcData(ifproc_file)
      file_data.append(file_data_1)
      obsNums.append(obsN)
      peaks.append(peak)
      image_files.append(image_file)
    print(obsNums)
    print(peaks)
    print(image_files)
    allfocus_results_d = allfocus(obsNums, peaks, image_files, file_data, opt=opt, row_id=None, col_id=None)
    image_file = allfocus_results_d['png']
    params = allfocus_results_d['params']
    status = allfocus_results_d['status']
    msg = allfocus_results_d['msg']
    print(allfocus_results_d)

if __name__ == '__main__':
    opt = 0
    obsNumList = list(range(94823, 94832+1, 1))
    obsNumList = list(range(99880, 99883+1, 1))
    obsNumList = list(range(97764, 97767+1, 1))
    obsNumList = list(range(110020, 110026+1, 1))
    obsNumList = list(range(110010, 110015+1, 1))
    obsNumList = list(range(110372, 110377+1, 1))

    if len(sys.argv) > 1 and sys.argv[1].startswith('0x'):
        opt = int(sys.argv[1], 0)
        print ('opt =', opt)
    elif len(sys.argv) > 2:
        try:
           opt = int(sys.argv[1], 0)
           print ('opt =', opt)
        except:
           pass
    try:
        if not sys.argv[-1].startswith('0x'):        
            obsNumList = ast.literal_eval(sys.argv[-1])
    except Exception as e:
        print(e)
        print('using default ObsNumList', obsNumList)

    print(obsNumList)
    line_list = None
    baseline_list = None
    linefocus(obsNumList, opt=opt, line_list=line_list, baseline_list=baseline_list, baseline_fit_order=0, tsys=200, tracking_beam=None)
    import matplotlib.pyplot as pl
    pl.show()

