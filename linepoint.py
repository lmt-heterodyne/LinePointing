import numpy as np
from lmtslr.spec.spec import SpecBankData, SpecBankCal
from lmtslr.viewer.spec_viewer import SpecBankViewer, SpecCalViewer
from lmtslr.ifproc.ifproc import lookup_ifproc_file, IFProcQuick, IFProcData, IFProcCal
from lmtslr.utils.roach_file_utils import find_roach_from_pixel, lookup_roach_files
import sys
from beam import BeamMap
from beam_viewer import BeamMapView
from merge_png import merge_png
import time
import os
import matplotlib.pyplot as pl
from msg_image import mkMsgImage

def linepoint(args_dict, view_opt=0):

    obsnum = (args_dict.get('ObsNum', None))
    spec_cont = (args_dict.get('SpecOrCont', 'Spec')).lower()
    line_list = (args_dict.get('LineList', None))
    baseline_list = (args_dict.get('BaselineList', None))
    baseline_fit_order = (args_dict.get('BaselineFitOrder', 0))
    tsys = (args_dict.get('TSys', None))
    tracking_beam = (args_dict.get('TrackingBeam', None))
    opt = (args_dict.get('Opt', 0))

    print('args = ', obsnum, spec_cont, line_list, baseline_list, baseline_fit_order, tsys, tracking_beam, opt)
    

    roach_pixels_all = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

    # define time stamp
    file_ts = '%d_%d_%d'%(obsnum, int(time.time()*1000), os.getpid())

    # define roach_list, can be modified if tracking a specific pixel
    roach_list = [0,1,2,3]

    # read the ifproc file to get the data and the tracking beam
    ifproc_file = lookup_ifproc_file(obsnum)
    if not ifproc_file:
        txt = 'No ifproc files found for %d'%obsnum
        print (txt)
        mkMsgImage(pl, obsnum, txt=txt, im='lmtlp_%s.png'%file_ts, label='Error', color='r')
        return {'plot_file': 'lmtlp_%s.png'%file_ts}

    if view_opt == 0x1234:
        IData = IFProcData(ifproc_file)
        pl.clf()
        pl.plot(IData.azmap, IData.elmap)
        pl.xlabel('AzMap (")')
        pl.ylabel('ElMap (")')
        pl.title('%d %s el=%lf'%(obsnum,IData.date_obs,IData.elev))
        pl.savefig('%s_azelmap.png'%obsnum, bbox_inches='tight')
        return {'plot_file': '%s_azelmap.png'%obsnum}

    # probe the ifproc file for obspgm
    ifproc_file_quick = IFProcQuick(ifproc_file)

    # get obspgm
    obspgm = ifproc_file_quick.obspgm

    print ('obsnum', obsnum)
    print ('receiver', ifproc_file_quick.receiver)
    print ('obspgm', obspgm)
    print ('tracking_beam', tracking_beam)

    if obspgm == 'Cal':
        ICal = IFProcCal(ifproc_file)
        ICal.compute_calcons()
        ICal.compute_tsys()
        for ipix in range(ICal.npix):
            print ('Tsys[%2d] = %6.1f'%(ipix,ICal.tsys[ipix]))
        if ICal.receiver == 'Msip1mm':
            msip1mm_pixel_description = {0: 'P0_USB',
                                         1: 'P0_LSB',
                                         2: 'P1_LSB',
                                         3: 'P1_USB'}
            dirname = '/var/www/vlbi1mm/'
            filename = 'vlbi1mm_tsys.html'
            if os.path.exists(dirname):
                filename = dirname + filename
            with open(filename, 'w') as fp:
                for ipix in range(ICal.npix):
                    desc =  msip1mm_pixel_description.get(ipix)
                    val = ICal.tsys[ipix]
                    fp.write("%s %3.1f\n" %(desc, val))
                fp.write("ObsNum %d\n" %(obsnum))
                fp.write("Time %3.1f\n" %(ICal.time[0]))

        pl.figure(num=3,figsize=(6,6))
        pl.clf()
        x = ICal.time-ICal.time[0]
        legend = []
        for ipix in range(ICal.npix):
            legend.append('%2d %6.1f'%(ipix,ICal.tsys[ipix]))
            y = ICal.level[:,ipix]
            pl.plot(x,y,'.')
        pl.legend(legend,prop={'size': 10})
        pl.suptitle("TSys %s ObsNum: %d"%(ICal.receiver,ICal.obsnum))
        pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')

        if spec_cont == 'cont':
            merge_png(['lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
        else:
            files,nfiles = lookup_roach_files(obsnum)
            SCal = SpecBankCal(files,ICal)
            # create viewer
            SV = SpecCalViewer()
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_tsys(SCal)
            pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')

            # merge plots
            merge_png(['lmtlp_%s.png'%file_ts, 'lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
        
        params = np.zeros((1,1))
        params[0,0] = np.mean(ICal.tsys)
        return {'plot_file': 'lmtlp_%s.png'%file_ts, 'params' : params, 'ifproc_data': ICal}

    # not a Cal
    # not a Cal
    # not a Cal
    # not a Cal
    
    # open data file
    IData = IFProcData(ifproc_file)
    
    bs_beams = IData.bs_beams
    map_motion = IData.map_motion
    line_velocity = IData.velocity
    reduce_type = 2

    # specify beam for fitting and plotting
    selected_beam = 10

    # check if tracking a specific pixel to modify the roach list and the selected_beam
    if 'Msip1mm' not in IData.receiver:
        tracking_beam = None
    if tracking_beam == None:
        tracking_beam = IData.tracking_beam
    if tracking_beam != -1:
        selected_beam = tracking_beam
        if bs_beams != []:
            roach_list = []
            for beam in bs_beams:
                roach_beam = find_roach_from_pixel(beam)
                if roach_beam[0] not in roach_list:
                    roach_list += roach_beam
        else:
            roach_list = find_roach_from_pixel(tracking_beam)

    if False:
        roach_list = [0,1,2,3]
    # build the roach directory list
    roach_dir_list = [ 'roach%d'%i for i in roach_list]

    # build the pixel list
    if bs_beams != []:
        pixel_list = bs_beams
    elif tracking_beam == -1 or obspgm == 'On':
        pixel_list = sum([roach_pixels_all[roach_index] for roach_index in roach_list], [])
    else:
        pixel_list = [tracking_beam]

    #pixel_list = [selected_beam]
    if False:
        pixel_list = sum([roach_pixels_all[roach_index] for roach_index in roach_list], [])

    print ('tracking_beam', tracking_beam)
    print ('selected_beam', selected_beam)
    print ('roach_list', roach_list)
    print ('roach_dir_list', roach_dir_list)
    print ('pixel_list', pixel_list)
    print ('bs_beams', bs_beams)
    print ('map_motion', map_motion)
    print ('line_velocity', line_velocity)
    print ('line_freq', IData.line_rest_frequency)

    if spec_cont == 'cont':
        print ('continuum reduction')
        pass
    else:
        print ('spectral line reduction')
        # find the roach files
        files,nfiles = lookup_roach_files(obsnum,roach_dir_list)
        if not files:
            txt = 'No roach files found for %d in %s' % (obsnum,str(roach_dir_list))
            print (txt)
            mkMsgImage(pl, obsnum, txt=txt, im='lmtlp_%s.png'%file_ts, label='Error', color='r')
            return {'plot_file': 'lmtlp_%s.png'%file_ts}

    # build reduction parameters
    #line_list = [[-27.5,-25.5]]

    # get line_list from arg then from ifproc data file then from default
    if not line_list or (all(isinstance(x, list) for x in line_list) and not [item for sublist in line_list for item in sublist]):
        line_list = IData.line_list
        print ('line_list in file', line_list)
        if not line_list:
            line_list = [[-30+line_velocity,30+line_velocity]]
            print ('line_list from default', line_list)
        else:
            print ('line_list from ifproc', line_list)
    else:
        if not any(isinstance(el, list) for el in line_list):
            print ('modify line_list to be a list of lists')
            line_list = [line_list]
        print ('line_list from arg', line_list)
        
    # get baseline_list from arg then from ifproc data file then from default
    if not baseline_list or not [item for sublist in baseline_list for item in sublist]:
        baseline_list = IData.baseline_list
        print ('baseline_list in file', baseline_list)
        if baseline_list:
            flat_list = [item for sublist in baseline_list for item in sublist]
        else:
            flat_list = None
        if not flat_list:
            baseline_list = [[-100-30+line_velocity,-30+line_velocity],[30+line_velocity,100+30+line_velocity]]
            print ('baseline_list from default', baseline_list)
        else:
            print ('baseline_list from ifproc', baseline_list)
    else:
        if not any(isinstance(el, list) for el in baseline_list):
            print ('modify baseline_list to be a list of lists')
            baseline_list = [baseline_list]
        print ('baseline_list from arg', baseline_list)

    if not tsys:
        tsys = 200

    if baseline_fit_order < 0:
        baseline_fit_order = 0
    elif baseline_fit_order > 3:
        baseline_fit_order = 3
        
    print ('line_list', line_list)
    print ('baseline_list', baseline_list)
    print ('baseline_fit_order', baseline_fit_order)
    print ('tsys', tsys)
    plot_axis = [-100-30+line_velocity, 100+30+line_velocity,-5,15] # modify y according to peak spectrum
    fit_circle = 30

    # create a list of cal spectra
    tsys_spectra = []

    if spec_cont == 'cont':
        pass
    else:
        # create the spec_bank object. This reads all the roaches in the list "files"
        SData = SpecBankData(files,IData,pixel_list=pixel_list)

        # set the pixel list to the pixels from the files we could find
        print ('unmodified pixel_list = ', pixel_list)
        pixel_list = SData.roach_pixel_ids
        print ('modified pixel_list = ', pixel_list)

    # check whether to use calibration and open necessary file
    use_calibration = True
    #use_calibration = False
    if use_calibration == True:
        calobsnum = IData.calobsnum
        print ('cal obs num = ', calobsnum)
        ifproc_cal_file = ''
        if calobsnum > 0:
            cal_files,ncalfiles = lookup_roach_files(calobsnum,roach_dir_list)
            ifproc_cal_file = lookup_ifproc_file(calobsnum)
        if ifproc_cal_file == '':
            use_calibration = False
        else:
            ICal = IFProcCal(ifproc_cal_file)
            if spec_cont == 'cont':
                ICal.compute_calcons()
                ICal.compute_tsys()
                for ipix in range(ICal.npix):
                    print ('Tsys[%2d] = %6.1f'%(ipix,ICal.tsys[ipix]))
            else:
                SCal = SpecBankCal(cal_files,ICal,pixel_list=pixel_list)
                check_cal = SCal.test_cal(SData)
                for ipix in range(SData.npix):
                    tsys_spectra.append(SCal.roach[ipix].tsys_spectrum)
                    print ('Tsys[%2d] = %6.1f'%(ipix,SCal.roach[ipix].tsys))
                if check_cal > 0:
                    print ('WARNING: CAL MAY NOT BE CORRECT')

    # line statistics
    line_stats_all = []

    if spec_cont == 'cont':
        if use_calibration == False:
            IData.dont_calibrate_data()
        else:
            IData.calibrate_data(ICal)
        IData.create_map_data()

    else:
        if use_calibration == False:
            print ('WARNING: NOT USING CAL, USING TSYS',tsys)
            for ipix in range(SData.npix):
                tsys_spectra.append(0)

        SData.cal_flag = use_calibration
        # reduce all spectra
        if obspgm == 'Map' or obspgm == 'Lissajous':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_spectra(stype=2,calibrate=use_calibration,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)
        elif obspgm == 'Bs':
            SData.roach[0].reduce_ps_spectrum(stype=2,normal_ps=False,calibrate=use_calibration,tsys_spectrum=tsys_spectra[0],tsys_no_cal=tsys)
            SData.roach[1].reduce_ps_spectrum(stype=2,normal_ps=True,calibrate=use_calibration,tsys_spectrum=tsys_spectra[1],tsys_no_cal=tsys)
        elif obspgm == 'Ps':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_ps_spectrum(stype=2,normal_ps=True,calibrate=use_calibration,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)
        elif obspgm == 'On':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_on_spectrum(calibrate=False,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)

        # set the baseline channels from velocities
        SData.make_velocity_list(baseline_list,'baseline')
        # set the line integration channels from velocities
        SData.make_velocity_list(line_list,'line')

        # create viewer
        SV = SpecBankViewer()

        # bs
        if obspgm == 'Bs':
            theSpectrum = (SData.roach[0].ps_spectrum+SData.roach[1].ps_spectrum)/2.
            v = SData.create_velocity_scale()
            line_stats = SData.roach[0].LineStatistics(SData.roach[0],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[0],SData.roach_pixel_ids[1],obspgm)
            line_stats_all.append(line_stats)
            print ('line_stats =', line_stats.to_string())
            params = np.zeros((1,2))
            params[0,0] = line_stats.yint
            params[0,1] = pixel_list[0]
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_bs(SData,baseline_fit_order,plot_axis,line_stats)

        # on
        elif obspgm == 'On':
            v = SData.create_velocity_scale()
            params = np.zeros((SData.npix,2))
            for ipix in range(SData.npix):
                theSpectrum = SData.roach[ipix].on_spectrum
                line_stats = SData.roach[ipix].LineStatistics(SData.roach[ipix],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[ipix],SData.roach_pixel_ids[ipix],obspgm)
                line_stats_all.append(line_stats)
                print ('line_stats =', line_stats.to_string())
                params[ipix,0] = line_stats.yint
                params[ipix,1] = pixel_list[ipix]
                prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1]))

            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_on(SData)

        # ps
        elif obspgm == 'Ps':
            v = SData.create_velocity_scale()
            params = np.zeros((SData.npix,2))
            for ipix in range(SData.npix):
                theSpectrum = SData.roach[ipix].ps_spectrum
                line_stats = SData.roach[ipix].LineStatistics(SData.roach[ipix],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[ipix],SData.roach_pixel_ids[ipix],obspgm)
                line_stats_all.append(line_stats)
                print ('line_stats =', line_stats.to_string())
                params[ipix,0] = line_stats.yint
                params[ipix,1] = pixel_list[ipix]
                prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1]))

            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_ps(SData,baseline_fit_order,plot_axis,line_stats_all)

        # grid map
        elif map_motion == 'Discrete':
            # this does a baseline and integration for the pixels in the list
            SData.create_map_grid_data(SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,pixel_list=pixel_list)

            # plot the spectra
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_all_spectra(SData,selected_beam,plot_axis,SData.blist,SData.nb)

        # raster map
        elif map_motion == 'Continuous' or obspgm == 'Lissajous':
            # this does a baseline and integration for the pixels in the list
            SData.create_map_data(SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,pixel_list=pixel_list)

            # show the peak spectrum
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_peak_spectrum(SData,selected_beam,plot_axis,SData.blist,SData.nb)

        # save spectra plot
        pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')

        # write a combined ifproc/roach file
        #SW = spec_bank_writer()
        #spec_files = SW.write(ifproc_file,SData,roach_list=roach_list,pixel_list=pixel_list,tracking_beam=tracking_beam)

    # set grid spacing
    grid_spacing = 1
    # now use the BeamMap class to solve for peaks
    B = None
    if obspgm == 'Map' or obspgm == 'Lissajous':
        #pixel_list = [i for i in range(16)]
        print ('beam map pix list', pixel_list)
        if IData.receiver == "Msip1mm":
            fit_circle = 10
        else:
            fit_circle = 30
        if spec_cont == 'cont':
            B = BeamMap(IData,pix_list=pixel_list)
            B.fit_peaks_in_list(fit_circle=fit_circle)
        else:
            B = BeamMap(SData,pix_list=pixel_list)
            B.fit_peaks_in_list(fit_circle=fit_circle)
        if tracking_beam != -1:
            params = np.zeros((1,4))
            if len(pixel_list) == 1:
                pix_id = 0
            else:
                pix_id = B.BData.find_map_pixel_index(tracking_beam)
            params[0,0] = B.peak_fit_params[pix_id,0]
            params[0,1] = B.peak_fit_params[pix_id,1]
            params[0,2] = B.peak_fit_params[pix_id,3]
            params[0,3] = pixel_list[pix_id]
        else:
            params = np.zeros((len(pixel_list), 4))
            for ipix in range(len(pixel_list)):
                params[ipix,0] = B.peak_fit_params[ipix,0]
                params[ipix,1] = B.peak_fit_params[ipix,1]
                params[ipix,2] = B.peak_fit_params[ipix,3]
                params[ipix,3] = pixel_list[ipix]
        BV = BeamMapView()
        BV.print_pixel_fits(B)

        if map_motion == 'Discrete':
            # show a map of the data with peak position overlain
            BV.set_figure(figure=2)
            BV.open_figure()
            #BV.add_subplot(SV.fig)
            BV.map(B,[],grid_spacing,apply_grid_corrections=True)
            if spec_cont != 'cont':
                BV.show_peaks(B,apply_grid_corrections=True,show_map_points=selected_beam)
            pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')

            ###BV.map3d(B,[],grid_spacing,apply_grid_corrections=True)

            if spec_cont == 'cont':
                BV.set_figure(figure=10)
                BV.open_figure()
                BV.show_fit(B,selected_beam)
                pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')
            # merge plots
            merge_png(['lmtlp_%s.png'%file_ts, 'lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)

        else:
            # show a map of the data with peak position overlain
            BV.set_figure(figure=2)
            BV.open_figure()
            #BV.add_subplot(SV.fig)
            BV.map(B,[],grid_spacing,apply_grid_corrections=True)
            BV.show_peaks(B,apply_grid_corrections=True)
            pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')
            if spec_cont == 'cont':
                BV.set_figure(figure=10)
                BV.open_figure()
                BV.show_fit(B,selected_beam)
                pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')
            # merge plots
            merge_png(['lmtlp_%s.png'%file_ts, 'lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
    
    if view_opt & 0x2:
        if obspgm == 'Map' or obspgm == 'Lissajous':
            # show the fit to the data
            BV.set_figure(figure=10)
            BV.open_figure()
            BV.show_fit(B,selected_beam)

        if map_motion == 'Continuous':
            if spec_cont == 'cont':
                BV.set_figure(figure=11)
                BV.open_figure()
                BV.sanchez_map(B,[-100,100,-100,100],grid_spacing)
                BV.show_peaks(B,apply_grid_corrections=False)
            else:
                # the "sanchez map" shows the array grid on the sky
                SV.set_figure(figure=11)
                SV.open_figure()
                SV.sanchez_map(SData,[-100,100,-100,100],grid_spacing,None,pixel_list=pixel_list)

                # the "map" uses the array grid model to align the pixels
                SV.set_figure(figure=12)
                SV.open_figure()
                SV.map(SData,[-200,200,-200,200],grid_spacing,None,pixel_list=pixel_list)

                # show the waterfall plot near the peak spectrum
                SV.set_figure(figure=13)
                SV.open_figure()
                SV.waterfall(SData,selected_beam,[-1500,1500],[-1,10],SData.blist,SData. nb)

        pl.show()

    print ('plot_file', 'lmtlp_%s.png'%file_ts)
    print ('params', params)
    return {'plot_file': 'lmtlp_%s.png'%file_ts,
            'params': params,
            'ifproc_data': IData,
            'line_stats': line_stats_all,
            'ifproc_file': ifproc_file,
            'peak_fit_params': B.peak_fit_params,
            'peak_fit_errors': B.peak_fit_errors,
            'peak_fit_snr': B.peak_fit_snr,
            'clipped': B.clipped,
            'pixel_list': pixel_list
    }

if __name__ == '__main__':
    # define options, 0 = write spec file only, 1 = plot, 2 = generate grid
    opt = 0

    if len(sys.argv) > 2:
        try:
            opt = int(sys.argv[1], 0)
            print ('opt =', opt)
        except:
            pass

    # Chi_Cyg cal
    obsNum = 78003
    # Chi_Cyg grid
    obsNum = 78004
    # IRC otf
    obsNum = 76406
    # MARS otf
    obsNum = 77358
    # BS
    obsNum = 78060
    # PS
    obsNum = 76085
    # R-Cas Ra Map
    obsNum = 94052

    try:
        arg = sys.argv[-1]
        if arg.startswith('c'):
            obsNum = 78003
        elif arg.startswith('g'):
            obsNum = 78004
        elif arg.startswith('m'):
            obsNum = 76406
        elif arg.startswith('p'):
            obsNum = 76085
        elif arg.startswith('b'):
            obsNum = 78075
        elif arg == ('9pt'):
            obsNum = 78091
        else:
            obsNum = int(arg)
    except:
        pass

    if obsNum > 0:
        args_dict = dict()
        args_dict['ObsNum'] = obsNum
        args_dict['SpecOrCont'] = 'Cont' if opt & 0x1000 else 'Spec'
        args_dict['LineList'] = None
        args_dict['BaselineList'] = None
        args_dict['BaselineFitOrder'] = 0
        args_dict['TSys'] = None
        args_dict['TrackingBeam'] = None
        args_dict['Opt'] = opt
    
        
        lp_dict = linepoint(args_dict, view_opt=opt)
        print(lp_dict)

        # show plot
        if opt & 0x1:
            pl.show()
    else:
        print("obsnum %d not valid" % obsNum)
            

