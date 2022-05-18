import ast
import sys
import json
from lmtlp_reduce_cli import lmtlp_reduce_cli
from allfocus import allfocus

def linefocus(obsNumList, opt, line_list, baseline_list, baseline_fit_order, tsys, tracking_beam):
    obsNums = []
    peaks = []
    data_files = []
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
      peak = results_dict['pk']
      azOff = results_dict['x']
      elOff = results_dict['y']
      azUserOff = results_dict['az_user_off']
      elUserOff = results_dict['el_user_off']
      obsPgm = results_dict['obspgm']
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
    obsNumList = list(range(94823, 94832+1, 1))
    obsNumList = list(range(97764, 97767+1, 1))
    obsNumList = list(range(99880, 99883+1, 1))
    print(obsNumList)

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
        print('using default ObsNumList', obsNumList)

    line_list = None
    baseline_list = None
    linefocus(obsNumList, opt=opt, line_list=line_list, baseline_list=baseline_list, baseline_fit_order=0, tsys=200, tracking_beam=None)

